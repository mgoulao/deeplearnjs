/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as dl from '../index';
import {ALL_ENVS, describeWithFlags, expectArraysClose} from '../test_util';

describeWithFlags('AdagradOptimizer', ALL_ENVS, () => {
  it('basic', () => {
    const learningRate = .1;
    const initialAccumulatorValue = .1;
    const optimizer = dl.train.adagrad(learningRate, initialAccumulatorValue);

    const x = dl.tensor1d([1, 2]).variable();

    const f = () => x.square().sum() as dl.Scalar;

    let numTensors = dl.memory().numTensors;

    let cost = optimizer.minimize(f, /* returnCost */ true);

    // Cost & accumulator should be the only additional arrays.
    expect(dl.memory().numTensors).toBe(numTensors + 2);

    // epsilon = 1-e8
    // newAccumulatedGrad = accumulatedGrad + grad^2
    // x -= (learningRate * grad) / sqrt(newAccumulatedGrad + eps)
    // de/dx = [2, 4]
    // accumulatedGrad = [0.1, 0.1]
    // newAccumulatedGrad = [4.1, 16.1]
    // x = [0.9012270405, 1.900311042]
    expectArraysClose(x, [0.9012270405, 1.9003110428]);

    cost.dispose();
    numTensors = dl.memory().numTensors;

    cost = optimizer.minimize(f, /* returnCost */ false);

    // de/dx = [1.802454081, 3.9501555214]
    // accumulatedGrad = [4.1, 16.1]
    // newAccumulatedGrad = [7.3488407141, 31.7037286432]
    // x = [0.8347372764, 1.83015597828]

    // TODO: Fix numerical precision.
    expectArraysClose(x, [0.8347372764, 1.83015597828], 1e-2);

    // There should be no new additional Tensors.
    expect(dl.memory().numTensors).toBe(numTensors);

    expect(cost).toBe(null);

    x.dispose();
    optimizer.dispose();

    // The only tensor remaining is the argument to variable().
    expect(dl.memory().numTensors).toBe(1);
  });
});
