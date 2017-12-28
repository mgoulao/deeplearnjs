/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import {NDArrayMath} from './math';
import {DataTypes, NDArray, Rank, RankMap, Scalar} from './ndarray';

/** A node's activation function and its derivative. */
export interface ActivationFunction {
  output<R extends keyof Rank>(
      math: NDArrayMath,
      input: NDArray<keyof DataTypes, R>): RankMap<keyof DataTypes>[R];
  der<T extends NDArray>(math: NDArrayMath, input: T, output: T): T;
  dispose(): void;
}

export class TanHFunc implements ActivationFunction {
  private one = Scalar.new(1);

  output<R extends keyof Rank>(
      math: NDArrayMath,
      x: NDArray<keyof DataTypes, R>): RankMap<keyof DataTypes>[R] {
    return math.tanh(x);
  }

  der<T extends NDArray>(math: NDArrayMath, x: T, y: T) {
    return math.scope(() => {
      const ySquared = math.elementWiseMul(y, y);
      // 1 - y^2.
      return math.scalarMinusArray(this.one, ySquared);
    });
  }

  dispose() {
    this.one.dispose();
  }
}

export class ReLUFunc implements ActivationFunction {
  output<R extends keyof Rank>(
      math: NDArrayMath,
      x: NDArray<keyof DataTypes, R>): RankMap<keyof DataTypes>[R] {
    return math.relu(x) as RankMap<keyof DataTypes>[R];
  }

  der<T extends NDArray>(math: NDArrayMath, x: T, y: T) {
    return math.step(x);
  }

  dispose() {}
}

export class LeakyReluFunc implements ActivationFunction {
  private alpha: number;

  constructor(alpha: number) {
    this.alpha = alpha;
  }

  output<R extends keyof Rank>(
      math: NDArrayMath,
      x: NDArray<keyof DataTypes, R>): RankMap<keyof DataTypes>[R] {
    return math.leakyRelu(x, this.alpha) as RankMap<keyof DataTypes>[R];
  }

  der<T extends NDArray>(math: NDArrayMath, x: T, y: T) {
    return math.step(x, this.alpha);
  }

  dispose() {}
}

export class SigmoidFunc implements ActivationFunction {
  output<R extends keyof Rank>(
      math: NDArrayMath,
      x: NDArray<keyof DataTypes, R>): RankMap<keyof DataTypes>[R] {
    return math.sigmoid(x);
  }

  der<T extends NDArray>(math: NDArrayMath, x: T, y: T): T {
    return math.scope(() => {
      // y * (1 - y) = y - y^2
      const ySquared = math.elementWiseMul(y, y);
      return math.subStrict(y, ySquared);
    });
  }

  dispose() {}
}

export class SquareFunc implements ActivationFunction {
  private two = Scalar.new(2);

  output<R extends keyof Rank>(
      math: NDArrayMath,
      x: NDArray<keyof DataTypes, R>): RankMap<keyof DataTypes>[R] {
    return math.multiplyStrict(x, x) as RankMap<keyof DataTypes>[R];
  }

  der<T extends NDArray>(math: NDArrayMath, x: T, y: T) {
    // dy/dx = 2*x.
    return math.scalarTimesArray(this.two, x);
  }

  dispose() {
    this.two.dispose();
  }
}

export class EluFunc implements ActivationFunction {
  output<R extends keyof Rank>(
      math: NDArrayMath,
      x: NDArray<keyof DataTypes, R>): RankMap<keyof DataTypes>[R] {
    return math.elu(x);
  }

  der<T extends NDArray>(math: NDArrayMath, x: T, y: T) {
    return math.eluDer(x);
  }

  dispose() {}
}
