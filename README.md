# matchmaker
Estimate the skill of players based on past matches.

Accidentally rediscovered the [Bradley–Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) while playing with keras.

Cool thing about the neural net implementation is that it can model the likelihood function non-linearly and, as a bonus, you can even play with latent variables and try to understand what makes winners winners and losers losers. Too bad it needs tons of data though.
