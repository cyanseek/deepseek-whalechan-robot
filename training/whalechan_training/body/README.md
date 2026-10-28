# Experimental VD32 physical body

This is the project's parameterized dynamics candidate: 23 actuated joints,
147 named bodies, 30 body collision geoms and an estimated mass of 1.267 kg.
`robot.xml` contains primitive collision geometry and explicit inertias;
`contract.json` records joint names, home pose and estimated motor parameters.
The task adds a plane and its own actuators. It does not need the character
texture/skin download. The public character viewer continues to use VD24.

Masses, inertias and the speed-dependent torque envelope are engineering
estimates awaiting hardware measurements. The enlarged sole support, longer
legs and load-bearing tail are this project's simulation adaptations. Character
proportions derive from the existing visual delivery; see the repository's
[source credits](../../../README.en.md#credits-and-licenses) and
[asset terms](../../../docs/ASSETS.md).
