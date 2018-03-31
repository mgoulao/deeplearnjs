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

import * as dl from '../index';
// tslint:disable-next-line:max-line-length
import {ALL_ENVS, describeWithFlags, expectArraysClose} from '../test_util';
import {Rank} from '../types';

describeWithFlags('conv1d', ALL_ENVS, () => {
  it('conv1d input=2x2x1,d2=1,f=1,s=1,d=1,p=same', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 'same';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = dl.tensor3d([1, 2, 3, 4], inputShape);
    const w = dl.tensor3d([3], [fSize, inputDepth, outputDepth]);

    const result = dl.conv1d(x, w, stride, pad, dataFormat, dilation);

    expect(result.shape).toEqual([2, 2, 1]);
    expectArraysClose(result, [3, 6, 9, 12]);
  });

  it('conv1d input=4x1,d2=1,f=2x1x1,s=1,d=1,p=valid', () => {
    const inputDepth = 1;
    const inputShape: [number, number] = [4, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 'valid';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = dl.tensor2d([1, 2, 3, 4], inputShape);
    const w = dl.tensor3d([2, 1], [fSize, inputDepth, outputDepth]);

    const result = dl.conv1d(x, w, stride, pad, dataFormat, dilation);

    expect(result.shape).toEqual([3, 1]);
    expectArraysClose(result, [4, 7, 10]);
  });

  it('conv1d input=4x1,d2=1,f=2x1x1,s=1,d=2,p=valid', () => {
    const inputDepth = 1;
    const inputShape: [number, number] = [4, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const fSizeDilated = 3;
    const pad = 'valid';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 2;
    const dilationWEffective = 1;

    const x = dl.tensor2d([1, 2, 3, 4], inputShape);
    const w = dl.tensor3d([2, 1], [fSize, inputDepth, outputDepth]);
    // adding a dilation rate is equivalent to using a filter
    // with 0s for the dilation rate
    const wDilated =
        dl.tensor3d([2, 0, 1], [fSizeDilated, inputDepth, outputDepth]);

    const result = dl.conv1d(x, w, stride, pad, dataFormat, dilation);
    const expectedResult =
        dl.conv1d(x, wDilated, stride, pad, dataFormat, dilationWEffective);

    expect(result.shape).toEqual(expectedResult.shape);
    expectArraysClose(result, expectedResult);
  });

  it('conv1d input=14x1,d2=1,f=3x1x1,s=1,d=3,p=valid', () => {
    const inputDepth = 1;
    const inputShape: [number, number] = [14, inputDepth];
    const outputDepth = 1;
    const fSize = 3;
    const fSizeDilated = 7;
    const pad = 'valid';
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 3;
    const dilationWEffective = 1;

    const x = dl.tensor2d(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], inputShape);
    const w = dl.tensor3d([3, 2, 1], [fSize, inputDepth, outputDepth]);
    // adding a dilation rate is equivalent to using a filter
    // with 0s for the dilation rate
    const wDilated = dl.tensor3d(
        [3, 0, 0, 2, 0, 0, 1], [fSizeDilated, inputDepth, outputDepth]);

    const result = dl.conv1d(x, w, stride, pad, dataFormat, dilation);
    const expectedResult =
        dl.conv1d(x, wDilated, stride, pad, dataFormat, dilationWEffective);

    expect(result.shape).toEqual(expectedResult.shape);
    expectArraysClose(result, expectedResult);
  });

  it('throws when x is not rank 3', () => {
    const inputDepth = 1;
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    // tslint:disable-next-line:no-any
    const x: any = dl.tensor2d([1, 2, 3, 4], [2, 2]);
    const w = dl.tensor3d([3, 1], [fSize, inputDepth, outputDepth]);

    expect(() => dl.conv1d(x, w, stride, pad, dataFormat, dilation))
        .toThrowError();
  });

  it('throws when weights is not rank 3', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const pad = 0;
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = dl.tensor3d([1, 2, 3, 4], inputShape);
    // tslint:disable-next-line:no-any
    const w: any = dl.tensor4d([3, 1, 5, 0], [2, 2, 1, 1]);

    expect(() => dl.conv1d(x, w, stride, pad, dataFormat, dilation))
        .toThrowError();
  });

  it('throws when x depth does not match weight depth', () => {
    const inputDepth = 1;
    const wrongInputDepth = 5;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 2;
    const pad = 0;
    const stride = 1;
    const dataFormat = 'NWC';
    const dilation = 1;

    const x = dl.tensor3d([1, 2, 3, 4], inputShape);
    const w = dl.randomNormal<Rank.R3>([fSize, wrongInputDepth, outputDepth]);

    expect(() => dl.conv1d(x, w, stride, pad, dataFormat, dilation))
        .toThrowError();
  });

  it('throws when both stride and dilation are greater than 1', () => {
    const inputDepth = 1;
    const inputShape: [number, number, number] = [2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 'same';
    const stride = 2;
    const dataFormat = 'NWC';
    const dilation = 2;

    const x = dl.tensor3d([1, 2, 3, 4], inputShape);
    const w = dl.tensor3d([3], [fSize, inputDepth, outputDepth]);

    expect(() => dl.conv1d(x, w, stride, pad, dataFormat, dilation))
        .toThrowError();
  });
});
