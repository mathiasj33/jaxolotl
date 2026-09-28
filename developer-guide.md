# Developer guide

> This guide was generated with GPT-6-Astra and reviewed manually for correctness.

Jaxolotl separates an LTL-RL algorithm into task construction, environment
wrappers, a model, and an evaluation agent. Most algorithms share the same
training and evaluation loops. Adding an algorithm usually means implementing
these components and connecting them through configuration.

This guide describes the main abstractions and how the existing implementations
fit together. See the [README](README.md) for installation and commands to run
experiments.

## Repository structure and configuration

| Location | Responsibility |
| --- | --- |
| [src/jaxolotl/alg](src/jaxolotl/alg) | Algorithm-specific curricula, task representations, wrappers, models, and evaluation agents. |
| [src/jaxolotl/rl](src/jaxolotl/rl) | RL optimizers, the actor–critic interface, and action distributions. |
| [src/jaxolotl/environments](src/jaxolotl/environments) | Base environments, observation and action specifications, resets, and general-purpose wrappers. |
| [src/jaxolotl/ltl](src/jaxolotl/ltl) | Formula parsing and progression, automaton construction, and reach-avoid path search. |
| [src/jaxolotl/networks](src/jaxolotl/networks) | Reusable observation, set, sequence, and graph encoders. |
| [src/jaxolotl/eval](src/jaxolotl/eval) | Evaluation agent interface, rollout loop, and model loading. |
| [src/jaxolotl/eqx_utils](src/jaxolotl/eqx_utils) | Equinox/JAX utilities for batching, control flow, training state, and serialization. |
| [scripts](scripts) | Entry points for training, preprocessing, evaluation, and visualization. |
| [conf](conf) | Hydra configuration connecting the implementations and defining experiments. |

The configuration distinguishes the **LTL-RL algorithm** (`alg`, such as
`struct_ltl`) from the **RL optimizer** (`rl_alg`, such as `ppo`). In
[conf/alg](conf/alg), an algorithm supplies four main hooks:

- `wrap_env`: construct the training or evaluation wrapper stack.
- `batcher`: convert curriculum samples into a batched JAX representation.
- `preprocess_formulas`: prepare evaluation formulae.
- `agent`: construct the evaluation agent around a trained model.

The matching configuration in `conf/experiment/<env>/<alg>.yaml` selects the
curriculum stage factory, model architecture, and training hyperparameters.
For example, [the Warehouse StructLTL experiment](conf/experiment/warehouse/struct_ltl.yaml)
connects a Boolean sequence curriculum, a StructLTL model, a composite actor,
and PPO. Hydra resolves `_target_` entries to Python callables; there is no
separate registry of LTL-RL algorithm classes.

## Python preprocessing and JAX execution

Task construction runs in ordinary Python before the compiled training or
evaluation loop. It can use strings, sets, dataclasses, variable-length lists,
automaton objects, and external tools such as Rabinizer or SemML. The rollout
loop instead operates on JAX-compatible PyTrees: nested structures whose dynamic
leaves are arrays.

The preprocessing boundary is where variable-size tasks become padded arrays,
transition tables, masks, and lengths. Tasks in a batch must have compatible
PyTree structures, shapes, and dtypes. During a compiled rollout, task progress
changes array values rather than the structure or size of the task. Keep Python
parsing, automaton construction, and disk access on the preprocessing side of
this boundary.

Models, environment states, and many task representations are `eqx.Module`
objects. Equinox transformations such as `filter_jit` and `filter_vmap` let the
framework compile and vectorize array computations while retaining static
configuration. Randomness is explicit: runtime methods receive JAX random keys
and return new state instead of mutating an environment in place.

## Environments and wrappers

The [Environment interface](src/jaxolotl/environments/environment.py) separates
physical dynamics from task semantics. Its central operations are:

```python
state, observation = env.reset(key, previous_state, params, options)
transition = env.step(key, state, action, params)
```

`reset` accepts a previous state because wrappers may retain information across
episodes, such as the next curriculum stage. The main data types are:

- `EnvParams`: static environment configuration, including the episode horizon.
  Changing these parameters can require recompilation.
- `EnvObservation`: an object with a `features` field containing a named tuple of
  environment-specific arrays. Task wrappers extend it with their own fields.
- `EnvTransition`: the next state and observation, reward, `terminated`,
  `truncated`, `terminal_observation`, propositions, and an `info` dictionary.
  Its `done` property is the disjunction of termination and truncation.

[ObservationSpec](src/jaxolotl/environments/observation_spec.py) describes the
shapes and dtypes of the physical observation fields. It is used to build
encoders without sampling an observation. [Action spaces](src/jaxolotl/environments/spaces.py)
can be discrete, continuous (`Box`), or composite, combining continuous and
discrete actions.

Each environment defines an ordered `propositions` tuple and the possible
Boolean assignments through `assignments()`. `compute_propositions` returns the
indices of currently true propositions, padded with `-1`; it does not return a
Boolean vector. `map_assignment_to_index` maps this representation to the
assignment index used by task transition tables. Preprocessing and runtime must
agree on proposition and assignment ordering.

[EnvWrapper and WrapperState](src/jaxolotl/environments/wrappers/wrapper.py)
provide composition. A stateful wrapper stores the inner state in `state` and
adds its own fields. Attribute forwarding exposes inner environment properties
and state fields through the stack. Wrappers update observations, rewards, or
termination while preserving the transition fields needed by outer wrappers.
In particular, `terminal_observation` must include the same task information as
an ordinary observation: PPO uses it to bootstrap at time-limit truncations,
even if automatic reset has already replaced `observation`.

Environment construction goes through [jaxolotl.make](src/jaxolotl/environments/registration.py),
which returns `(env, params)`. Reset generation is split into `sample_reset`
and `materialize`: a compact reset descriptor can be stored and later expanded
into a full simulator state. `reset_source="train"` or `"test"` installs a
`PrecomputedResetWrapper`; `"native"` samples resets directly. This keeps
expensive reset generation outside the rollout loop when a precomputed pool is
used. See [precompute_resets.py](scripts/precompute_resets.py).

## Training

### Curricula, preprocessing, and batching

The [curriculum interfaces](src/jaxolotl/alg/curriculum/curriculum.py) deliberately
do not prescribe a Python task type. A `Sampler[TSample]` implements
`sample(rng: random.Random) -> TSample`. A `CurriculumStage` samples tasks and
specifies a success threshold; `RandomCurriculumStage` wraps a sampler, and
`MultiRandomStage` mixes samplers with different probabilities.

Examples of `TSample` include formula strings for LTL2Action and SemLTL,
`ReachAvoidSequence` objects for DeepLTL, Boolean sequences for StructLTL,
reach-avoid subgoals for GenZ-LTL, and proposition names for GCRL-LTL. Training
samples therefore need not be complete LTL formulae.

A `SampleBatcher[TSample, TJaxSample]` implements:

```python
batch(samples: list[TSample], env, num_parallel: int = 1) -> TJaxSample
```

It performs any expensive task-specific preprocessing and returns a PyTree with
a leading sample axis on its dynamic array leaves. For example,
[ReachAvoidSequenceBatcher](src/jaxolotl/alg/deep_ltl/batching.py) encodes assignment
sets and sequence lengths, whereas [FormulaClosureBatcher](src/jaxolotl/alg/ltl2action/batching.py)
constructs formula progression closures and syntax graphs. Padding values and
masks belong to the representation; simply padding every field with zero is
not generally correct. [pad_and_stack](src/jaxolotl/eqx_utils/batching.py) is a
shared utility when corresponding array leaves can be padded independently.

`Curriculum` samples a fixed number of tasks per stage before training, batches
all stages together, and reshapes the sample axis into
`(num_stages, num_samples, ...)`. Runtime sampling then only selects a stage and
sample index. Batching stages together ensures that advancing to a harder stage
does not change array shapes. Even algorithms without progressive curriculum
learning use this task-pool machinery with a single stage.

[wrap_for_training](src/jaxolotl/alg/curriculum/factory.py) constructs the
configured stages and batcher, optionally adds the task wrapper, and installs
`CurriculumWrapper`. It loads a precomputed curriculum from
`data/<environment name>/<algorithm name>/curriculum.eqx` when available;
otherwise, it builds the samples in memory. [precompute_curriculum.py](scripts/precompute_curriculum.py)
uses the same machinery to save them in advance. Regenerate this artifact when
changing task semantics or encoding: loading checks stage and sample counts,
not every aspect of the preprocessing implementation.

At reset, `CurriculumWrapper` samples a task and passes it inward as
`CurriculumResetOptions(task=...)`. The shared
[CurriculumManager](src/jaxolotl/alg/curriculum/curriculum_manager.py) tracks
success across the parallel environments of each training seed. It advances the
population's frontier after the configured success window, threshold, and
coverage requirements are met. Individual environments adopt later stages
probabilistically, with the adopted stage taking effect at reset.

### Task wrappers and rewards

Each algorithm's `wrap_env(env, cfg, training)` hook defines how the base
environment acquires task semantics. The training task wrapper reads
`options.task`, stores task progress, adds task information to observations,
and computes rewards and termination from the environment's propositions.

For example, DeepLTL's [SequenceWrapper](src/jaxolotl/alg/deep_ltl/wrappers/sequence_wrapper.py)
advances a reach-avoid sequence when the current reach set is satisfied. It
returns `+1` on sequence completion, `-1` on an avoid violation, and `0`
otherwise; completion or violation ends the task. StructLTL reuses this wrapper
with a different sequence encoding. GenZ-LTL's subgoal wrapper also reports a
cost signal used by its constrained optimizer.

The stock curriculum wrapper reports success as `transition.reward > 0`, and
the manager uses that signal for completed episodes. A new reward scheme must
remain compatible with this convention or adapt the success reporting.

Several algorithms support **epsilon actions**: transitions that advance the
automaton or sequence without advancing the physical environment. Their action
distributions and wrappers agree on an extended action format, such as
`(env_action, epsilon_action)`, and an observation mask describes which epsilon
choices are legal. This is separate from the base environment's action space.

### Training loop

[scripts/train.py](scripts/train.py) builds the following stack, shown from
innermost to outermost:

```text
base environment / optional precomputed resets
  → TimeLimitWrapper
  → alg.wrap_env(..., training=True): task + curriculum wrappers
  → AutoResetWrapper(FULL)
  → LogWrapper
  → VectorizeWrapper
```

An algorithm can add other wrappers inside its hook; SemLTL also normalizes
training rewards. `AutoResetWrapper` starts a new episode when a transition is
done, while preserving its reward, termination flags, and terminal observation.
`VectorizeWrapper` runs the single-environment API across a batch of random keys
and states.

The script builds one model per seed, constructs the curriculum manager and
configured `RLAlgorithm`, then vectorizes training over seeds and compiles it.
[PPO](src/jaxolotl/rl/ppo.py) collects rollouts, updates curriculum progress,
computes generalized advantage estimates, and optimizes minibatches over
multiple epochs. [PPOSafe](src/jaxolotl/rl/ppo_safe.py), used by GenZ-LTL, adds
cost values and a learned Lagrangian multiplier. Logging and checkpointing use
callbacks out of the compiled loop.

## Model architecture

Models implement [ActorCritic](src/jaxolotl/rl/actor_critic.py). The usual pattern
is to encode the physical observation and task separately, combine their
features, and feed them to actor and critic heads:

```text
observation.features → environment encoder ─┐
                                            ├→ combined features → actor distribution
observation task fields → task encoder ─────┘                    → scalar value
```

The main extension point is `_compute_common_features(obs)`. The public
`model(obs)` returns an action distribution and value; `get_action(obs)` and
`get_value(obs)` expose the heads separately. These methods expect batched
observations. Individual [ObservationEncoder](src/jaxolotl/networks/observation_encoder.py)
instances encode one observation, and models apply them with `vmap`.
`FlattenMLPEncoder`, `FieldConvEncoder`, and `ParallelEncoder` support flat,
grid, and combined observation processing.

The actor is selected in configuration to match the environment's discrete,
continuous, or composite action space. `_get_actor_context(obs)` supplies extra
information such as epsilon masks. Task encoders reuse modules such as DeepSets,
GRUs, attention, and graph convolutions from `networks`.

The common architecture admits specialized models: GenZ-LTL conditions the
physical observation through an observation reduction function before encoding
it, and adds cost and Lagrangian heads. SemLTL supplies features for both the
current automaton state and candidate epsilon successors to its specialized
actors.

## Evaluation

Evaluation starts from LTL formula strings, even when training used simpler
subtasks. [scripts/eval/eval.py](scripts/eval/eval.py) constructs the environment,
calls `wrap_env(..., training=False)`, vectorizes it, preprocesses formulae,
loads models, and instantiates agents. There is a time-limit wrapper, but no
curriculum or automatic reset wrapper: evaluation runs a fixed set of episodes
and stops counting each episode when it completes.

### Formula preprocessing and evaluation wrappers

`preprocess_formulas(formulas, env)` returns algorithm-specific task data with a
leading formula axis. The evaluator selects one task and passes it to the
wrapper as `EvalResetOptions(task=...)`, shared across the parallel episodes.
This representation need not match the training batcher's output.

For automata-based reach-avoid algorithms, the
[shared preprocessing pipeline](src/jaxolotl/alg/common/reach_avoid/preprocessing.py)
constructs a limit-deterministic Büchi automaton (LDBA), prunes it against the
environment's assignments, handles forced epsilon transitions and sink states,
and enumerates candidate reach-avoid paths from each automaton state. Each
algorithm encodes those candidates in its own format. The result contains both
a batched `JaxLDBA` transition table and the candidate task representations.

An evaluation wrapper tracks progress against the complete formula. For
example, DeepLTL and StructLTL use
[LDBAWrapper](src/jaxolotl/alg/deep_ltl/wrappers/ldba_wrapper.py), which updates the
automaton state and reports accepting transitions and sink violations. It does
not use the training sequence-completion reward. LTL2Action instead progresses
a precomputed formula closure; SemLTL directly tracks its semantically labelled
LDBA. Finite/infinite evaluation semantics are configured through `cfg.eval.finite`;
LTL2Action also receives that flag during formula preprocessing.

### Agents and rollout

The [Agent interface](src/jaxolotl/eval/agent.py) separates action selection from
evaluation-time planning:

- `instantiate(model, ...)` constructs the agent.
- `update(obsv, state, props, env)` returns an updated agent after reset and after
  every step. Stateful planners store their selected task and other runtime
  state on the agent.
- `get_action(obsv)` returns a distribution; the evaluator samples from it or
  takes its mode for deterministic evaluation.

The default agent forwards observations to the model and needs no planning
state. DeepLTL's agent scores candidate sequences with the critic and selects a
new sequence when the automaton state changes. It then constructs the sequence
observation expected by the trained policy. StructLTL reuses that agent, while
GenZ-LTL and GCRL-LTL implement their own selection logic.

[Evaluator](src/jaxolotl/eval/eval.py) runs episodes in parallel, updating agents
inside the compiled loop. It records positive rewards, discounted returns,
lengths, and sink violations. An evaluation wrapper must provide
`info["is_sink"]`, and its rewards must have the intended evaluation meaning;
arbitrary training reward shaping would change the reported metrics.
The surrounding [evaluation utilities](src/jaxolotl/eval/utils.py) batch models
and formulae to control memory use.

## How the existing algorithms fit together

| Algorithm | Training representation and model | Evaluation |
| --- | --- | --- |
| [LTL2Action](src/jaxolotl/alg/ltl2action) | Formula strings become progression closures with precomputed assignment transitions and syntax graphs. An RGCN encodes the current formula graph. | Uses the formula-closure wrapper and default agent; progression becomes table lookup during rollout. |
| [DeepLTL](src/jaxolotl/alg/deep_ltl) | Reach-avoid sequences encode sets of assignments. Assignment embeddings and DeepSets encode each reach/avoid pair, then a GRU encodes the sequence. | An LDBA wrapper tracks the formula; the agent uses learned values to select candidate sequences. |
| [StructLTL](src/jaxolotl/alg/struct_ltl) | Boolean reach-avoid sequences encode propositions, negations, and clauses. Nested set encoders and a configurable sequence encoder produce task features. | Reuses DeepLTL's wrapper and agent with Boolean sequence preprocessing. `tokenized_ltl` and `gcn_ltl` supply alternative encodings and models. |
| [GenZ-LTL](src/jaxolotl/alg/genz_ltl) | Individual reach-avoid subgoals condition an observation reduction function. The model has reward, cost, and Lagrangian heads and trains with PPOSafe. | Uses an LDBA subgoal wrapper and an agent that selects subgoals, including switching after a timeout. |
| [GCRL-LTL](src/jaxolotl/alg/gcrl_ltl) | Proposition-conditioned policies and values use a goal embedding. A separate goal-conditioned value function (GCVF) is trained after PPO. | The agent uses the GCVF to score sequences and filters actions using learned values for unsafe goals. It requires discrete actions; entry points request environment discretization. |
| [SemLTL](src/jaxolotl/alg/sem_ltl) | Formula strings become SemML LDBAs carrying semantic embeddings. The model projects those embeddings and combines them with environment features, including features for epsilon successors. | Uses the semantic LDBA wrapper and default agent, with specialized actors that choose among multiple epsilon successors. |

## Adding an algorithm

Start with the existing algorithm closest to the intended task representation.
An implementation can reuse a wrapper or agent without inheriting its model;
StructLTL's configuration is a useful example of this composition.

1. **Define training tasks and their array representation.** Add samplers and a
   stage factory under `src/jaxolotl/alg/<name>/`. Implement a `SampleBatcher`
   that converts arbitrary Python samples into consistently shaped task data.
   Define padding, masks, and task progress operations together.
2. **Implement training task semantics.** Add a wrapper that accepts
   `CurriculumResetOptions.task`, initializes task state, augments observations,
   and produces rewards and termination. Expose it through
   `wrap_env(env, cfg, training)` and use `wrap_for_training` to install the
   curriculum. A single-stage curriculum is sufficient when staged learning is
   unnecessary; the stock trainer expects `env.curriculum`.
3. **Implement the model.** Extend `ActorCritic`, encode the wrapper's task
   fields, and configure an actor compatible with the wrapper's action format.
   The model factory receives `obs_spec`, `act_space`, `num_assignments`,
   `num_propositions`, `env_params`, and `key`; existing constructors accept
   unused configuration through `**kwargs`. Reuse PPO unless the learning rule
   itself needs to change.
4. **Define evaluation from full formulae.** Implement `preprocess_formulas`,
   the evaluation branch of `wrap_env`, and an agent if task selection requires
   planning. Keep the evaluation wrapper responsible for formula progress and
   metric semantics. The agent should translate its choices into observations
   understood by the trained model.
5. **Connect the configuration.** Add `conf/alg/<name>.yaml` with the four hooks
   and `conf/experiment/<env>/<name>.yaml` for each supported environment. The
   latter supplies the model, curriculum, and optimizer settings. Most additions
   require no changes to the entry points, but extra training phases or model
   artifacts do: GCRL-LTL's GCVF is an existing example.
6. **Validate the boundaries.** Test task conversion and padding, task progress
   and reward semantics, and batched model outputs. Check reset and step under
   JIT/vectorization, then run a small training and evaluation experiment. Include
   different task sizes and termination versus truncation; exercise epsilon
   actions when supported. Prefer tests of these behaviors over tests that only
   assert which internal functions were called.

To add a base environment, implement the abstract methods in `Environment`,
provide its state, parameters, structured observation specification, proposition
assignments, action space, and renderer, and register it in `environments/registration.py`.
Provide a compact reset descriptor and `materialize` implementation if storing
full reset states would be expensive. Add environment and experiment configs;
algorithms with environment-specific curricula or observation reduction may
need additional implementations too.

Use Python 3.12 through Pixi in this workspace. Keep library code in `src/jaxolotl`,
runnable utilities in `scripts`, and one-off experiments in `local`. Tests are
usually co-located with implementation files and named `*_test.py`. For example:

```bash
pixi run pytest --ignore=runs src/jaxolotl/alg/curriculum/curriculum_test.py
pixi run ruff check src/jaxolotl/alg/<name>
pixi run ruff format src/jaxolotl/alg/<name>
```
