[//]: # (A Model of Unsupervised Rationality)

Unsupervised utility-maximization for altruistic intelligence
===================================

To implement humanity's higher motives, it may be essential to model the outside world in a way that is scrutable to one's utility function (rather than e.g. relying on reward input).  The successful definition of such an agent may also have implications for epistemology, as it must have an answer for what to do e.g. when tricky anthropic questions are relevant.  It may even help to change the way people conceptualize rationality generally, from reward maximization or the "selfish gene" to something more consistent with intuitive morality.

The following is a mathematical definition of what it means exactly to be a utility-maximizing agent.  E.g. it may be used to verify whether an algorithm results in expected utility maximizing choices.  I will refer to it as the Hazelflax (for lack of a better name).  It seems to deal well with every case I have tested (including anthropics paradoxes).  It has the added benefits that:
1) it requires only:
    - a prior on the underlying universe
    - the raw data/self-knowledge available.  (It does not require (additional techniques to compute) the probabilities implied by evidence.)
2) it’s utility function is defined on the block universe. (It doesn’t require a reward input.) (I.e. it is “unsupervised”.)  This may allow it to serve the interests of life generally.  E.g. by seeding other planets with microorganisms that will eventually result in rational agents.

That is to say, it is an all-in-one world improving algorithm.

# Abstract idea

The basic idea is that you want to find an expected utility (EU) maximizing policy for agents that are identical to you in their decision procedure tho not necessarily in their observations.  The policy is defined as a function from one’s observations to a choice.  (It won’t do to just find the best choice given one’s observations since that may depend on the choices that identical agents make when they have different observations.)
(In this system, an “agent” is just a moment when a choice is made, thereby avoiding superfluous distinctions about identity across time.  An agent has observations and a program that maps them to a choice. )
You can calculate EU given a policy using a prior on universes.  These universes need to include enough detail to compute a full block universe just given a policy.  I.e. (the laws + initial conditions + randomness) + policy = history including everything we care about.
(For agents that are not identical, a universe must imply their code which we can execute to determine their choices.  When a universe presents an agent with our code, we can’t simply run it because that would cause an infinite regress.  Use the policy to compute its choice. )

Imagine you’re in the primordial casino, rolling the dice over and over, each time choosing a `universe` (according to `prior`).  Each time, you use your `policy` to unfold the `universe` into a block universe and apply your `utility` function, then add the resulting value to the tally.  The long-run average of this process is the expected utility given your `policy`.  (Hazelflax calculates this by taking the `prior` weighted sum of block universe utilities, across all universes.)

> Expected utility corresponding to the optimum policy:\
> `\max_{h \in H} \sum_{u \in U} P(u)utility(block(u, h))`
> - `H` is the set of candidate policies.
> - `U` is the set of possible universes. (Laws + randomness.  Everything needed to determine a block universe except your policy.)
> - `P(u)` is the prior probability of universe `u`.
> - `block(u,h)` is the block universe that unfolds from universe `u`, when agents with your code use policy `h`.  

# Examples

## [Sleeping beauty] [1]

The prior gives 0.5 probability to the universe where the coin comes up heads.  Same for tails.  For whatever utility function you want to maximize, just consider the implications of each candidate policy in each universe and apply the formula.

## [Doomsday paradox] [2]

The confusion here seems to be that the universe doesn’t make observers by pulling them from an urn or something.  It makes observers the same way it makes everything else.  Without further information, let’s assume the characteristics of the observers are independent and identically distributed, such that each one has the same probability of being identical to me.  This probably doesn’t change if the number of observers changes.

Bostrom Step 1: Suppose our prior has many universes with 100 cubicles, each with random agents in the cubicles.  My expected utility only depends upon my policy through the universes that have an exact match.  If an exact match is unlikely enough that none have two agents identical to me, of the universes that have a match, the following fraction have a match in a blue room:

blue match fraction / any match fraction = 
lim_{p->0} (1-(1-p)^90)/(1-(1-p)^100) = 0.9

Where p is the probability of a given agent matching.

Bostrom Step 2: The probability of the person in cubicle \#7 being exactly like me is unaffected by the coin flip.  So, given the evidence there is still a 50% probability of each coin outcome.  The Doomsday argument is invalid.  (The problem with the self-sampling assumption seems to be that the probability of any given observer being exactly like you doesn’t necessarily have anything to do with the total number of observers.)

[1]: <https://www.scientificamerican.com/article/why-the-sleeping-beauty-problem-is-keeping-mathematicians-awake/> "Why the ‘Sleeping Beauty Problem’ Is Keeping Mathematicians Awake"
[2]: <https://anthropic-principle.com/q=anthropic_principle/doomsday_argument/> "A Primer on the Doomsday Argument" 

# Formalization 

Suppose we want to make the choice with the best outcome.  I.e. that causes the probability distribution over block universes with the greatest expected utility.  How may we model the impact of a choice on the universe? (In contrast to e.g. Solomonov induction, which models an input stream.).  

Define a `universe` as a function:

    state -> choice -> Maybe (state, agent)

Where `state` is the current state of the universe, `choice` is the choice made by the last agent (whose turn it was to make a choice), and `(state, agent)` are the new state and next agent up to make a choice.  

Define an `agent` as: `(observations, code)`.
I.e. the observations an agent has at the time of considering a choice, the code it runs. 

(Note that this elides superfluous distinctions about continuity of identity, e.g. whether you’re the same person after destructive teleportation to Mars.  Every instance of choice-making is considered an agent.)

The agent’s `code` returns the agent’s choice, which is fed back to the `universe` function, allowing us to “unfold” the sequence of agents and their choices that constitute a block universe (on which the utility function is defined). 

    newtype block = [(state, agent, choice)]
    utility :: block -> real

Now if we just add a prior on universes, we can calculate the expected utility (EU) maximizing choice.  (Note that this prior only needs to apply to universes, avoiding the problems inherent in a prior on observations.)

    prior :: universe -> probability

## Code

The following is pseudo-code roughly based on Haskell language.

```haskell
class hazelflax choice state observations where
  newtype policy = observations -> choice
  -- an agent corresponds to a single choice. 
  -- code: code that compiles to a function of type policy (that always terminates)
  newtype agent = (observations, code)
  -- function that represents a candidate universe.  (Like the laws of physics but with agents surfaced so one’s utility function can be evaluated and the implications of one’s policy can be evaluated.)
  newtype universe = state -> choice -> Maybe (state, agent)
  -- block universe
  newtype block = [(state, agent, choice)]
  -- prior probability of a given model representing the real universe 
  prior :: universe -> probability
  -- utility function must be bounded
  utility :: block -> real
  -- possible universes (the domain of prior)
  universes :: set universe

  default_state :: state
  default_choice :: choice

  choose :: observations -> choice
  choose my_obs = policy_fun my_obs where
    -- Choose the best policy.  Since there are infinite policies and utilities are real valued, it is possible there is no maximum; really this should do something  more subtle, but I don't want to complicate it in a way that may distract from the main idea. 
    -- E.g. pick a $c \in choice s.t. !(\exist p \in policy. p obs != c and \all q \in policy. q obs == c => policy_ev p > policy_ev q)$.  I.e. there's no other choice with a corresponding policy that ranks higher than every policy that is consistent with this choice.
    policy_fun = max_{pcy \in policy} policy_ev pcy
  -- expected value of utility function given a policy (prior to observations)
  policy_ev :: policy -> real
  policy_ev policy_fun =
    sum $ map (\u -> prior u * utility (make_block u)) universes
    where
    make_block u = step default_state default_choice
      where
      step :: state -> choice -> block
      step s c =
        switch u s c
        | Just (s’, agent@(obs, code)) -> 
          (s’, agent, c’) : step s’ c’ where 
						-- MY_CODE = code of this program, perhaps instantiated Quine-style
            c’ = if code == MY_CODE
              then policy_fun obs
              else (compile_to_function code) obs
        | Nothing -> []

```

# Implementation 

`policy_ev` defines how good a policy is.  A policy might be implemented with minimax or a decision making algorithm yet to be invented.  To make it tractable, one may need to settle for some notion of the “logical expected value” of `policy_ev policy`.  With the question formalized, a sufficiently advanced automated theorem prover may be able to search for and verify policies. 

## Example: Ultimate conservationist

Suppose you want to maximize the probability that there is no last agent.  (I.e. utility is 1 if there is no last agent, 0 otherwise.)
First define a mapping from positive integers to policies and to `universe`s.
Define the prior of `universe` `n`, `1/2^n`.

For a given k, run the first k universes for at most k steps on each of the first k policies.  

For a given policy, adding up the priors on universes that haven't returned `Nothing` (plus the priors on the remaining universes that weren’t run) gives an upper-bound on its corresponding EU.

With a large enough k, the choice (given one’s observations) of the policy with the highest upper-bound is at least as good as any (assuming a bounded set of options for one’s choice).  

## Warflax

Maybe a war-game exercise could be modeled after the hazelflax:
Contestants submit policies and possible universes. Judges assess the probability of each universe and the utility of each universe-policy pairing, then calculate the expected utility given each policy. The contestant who submitted the highest EU policy wins.  (A universe may take random bits as a parameter, s.t. the EU of a policy given a universe can be estimated by sampling.)

(Do war-games often have flips that determine the nature of the universe? (E.g. what the weighting should be on subsequent coin flips.  E.g. how do untested pairings of weapon systems fare?) With an adversarial process for discovering the salient possibilities?)

(If this isn’t so different from traditional war-games, Hazelflax may be to war-games what a Turing machine is to a mathematician with a notebook. )

# Discussion 

Q: Why rely only on priors and raw evidence?

A: People seem to ask too much of probabilities.  Often the evidence doesn’t tell you an exact probability or even a probability range.  Translating the evidence to probabilities often entails losing information, yet decision making techniques often rely on probabilities.
Hazelflax skips the lossy intermediate epistemology and  may squeeze every drop of rational decision making from the evidence.

Q: Should `agent` also specify the runtime (and other computing resources) available so you only choose a policy that will return its choice in the time the universe makes available (in each circumstance)?

A: Probably.  (I didn't want to distract from the basic concept.)

Q: Why not represent other agents by their utility function (rather than their code)?

A: If you optimize over multiple utility functions and corresponding policies, how do you choose a point on the Pareto frontier?
I think the best you can do in practice is to model bounded rationality.
My intention for Hazelflax is a perfect agent (that may be uncomputable) that models the imperfections of other agents, so that as you approximate Hazelflax you take account of those imperfections (rather than having an imperfect agent that assumes other agents are perfect).  

Q: How can Hazelflax be approximated?
A: It specifies `policy_ev` as a mathematical function of the parameters and abstract methods of the typeclass (which may not be computable).  One may search the space of programs for a `policy` with a high `policy_ev`.  This may be possible to prove without computing `policy_ev`.  I.e. one might search for a `policy` that can be proven to have a higher `policy_ev` than your current `policy`.  Or one might use some notion of logical expected value to search for a `policy` that you expect to have a high `policy_ev`.
It won’t do to find a `choice` that can be proven to be compatible with a high `policy_ev` `policy`; an effective `choice` must be effective in the context of one’s bounded rationality.  I.e. your `policy` must return its `choice` using only the computing resources allowed by the `universe` every time it is invoked in `policy_ev` (e.g. time, concurrency, working memory, internal communication bandwidth).  E.g. you shouldn’t write a check given the high likelihood of there existing a policy that would give you a much higher income unless you know you can discover the policy in time. 

There’s surely no best decision making algorithm.  There may be different notions of approximation one can use to judge an algorithm against Hazelflax.  For any very complex set of candidate universes and notion of approximation, the Blum speedup theorem likely applies.

Q: Can Hazelflax model other agents with mixed strategies (or follow a mixed strategy when doing so is optimal)?

A: The optimal policy may be to flip a coin secretly, then (having the outcome in one’s observations) make a choice that depends on it.  From another agent’s perspective, there may be a candidate universe for each flip outcome (each having the same prior), s.t. the optimum policy needs to hedge.  (If you don’t flip a coin, the pattern of your behavior may be reflected in other agents’ observations s.t. their optimum policy anticipates your choice.)

Q: Is this schema compatible with infinite universes?

A: A universe may be infinite spatially, while the `universe` function that iterates over choices needs only a finite state at any given iteration. (Imaging a 4-D space being filled by repeatedly spiraling over the surface of a cone (s.t. for any point the preceding light-cone is filled in before it is reached).)

Q: Instead of the Hazelflax, why not define the crucible of natural selection?  Humans can concieve of all sorts of possibilities that don't exist in our universe (e.g. transfinite sets, oracles).

A: You might get the same benefit just by limiting the algorithmic complexity of your policy.  It may not be as optimal for the particular universe(s) for which you have a prior, but it may generalize better.  (Just an intuition.)

Q: Is the prior generally less important when the evidence is stronger?

A: Yes. Stronger evidence reduces the set of universes where your choice effects utility.  E.g. if there’s only one possible universe with an agent with your observations, you would make the choice that's best in that universe, regardless of the universe's probability.
If your choice of prior only applies to agents with no less evidence (I.e. future instances of yourself.), s.t. none of them appear in other universes, your choice of prior won't effect your choices except maybe indirectly.  E.g. if it leaks out or effects your computing time.
More realistically, if all the matching universes have the same physics but different random bits, the same argument applies. 

Q: Is it causal or evidential?

A: I think it has the benefits of an evidential decision theory. 
For a `universe` to model agents like Newcomb’s demon it must separate out its simulation of you as an agent, s.t. in `policy_ev` there will be two `agent`s that choose a box, one of which causes the contents of the boxes.
If the demon needs to run you, and you run the demon, ether `universe` or the `agent` representing the demon will not halt as it is required to.

