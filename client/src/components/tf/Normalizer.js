import * as tf from '@tensorflow/tfjs';
import { assert } from '@tensorflow/tfjs-core/dist/util';

export default class Normalizer {

  constructor(tensor) {
    this.tensor = this._normalize(tensor);
  }

  /**
   * Get the normalized tensor.
   * @returns tensor
   */
  getTensor() {
    return this.tensor;
  }

  /**
   * Dispose normalized tensor
   */
  dispose() {
    this.tensor.dispose();
  }

  /**
   * Normalize data
   * @param {tf.tensor} tensor
   * @returns tensor
   */
  normalize (tensor) {
    return this._normalize(tensor);
  }

  /**
   * Denormalize tensor
   * @param {tf.tensor} tensor
   * @param {number} optional dimension to denormalize
   */
  denormalize(tensor, dim) {
    if(dim !== undefined) {
      const denormalizedTensor = tensor.mul(this.max[dim].sub(this.min[dim])).add(this.min[dim]);
      return denormalizedTensor;
    }
    else if (this.min.length > 1) {
      // More than one feature
      // Split into separate tensors
      const features = tf.split(tensor, this.min.length, 1);

      // Denormalize
      const denormalized = features.map((featureTensor, i) => this._minMaxDenormalize(featureTensor, this.min[i], this.max[i]));

      const returnTensor = tf.concat(denormalized, 1);
      return returnTensor;
    }
    else {
      const denormalizedTensor = this._minMaxDenormalize(tensor, this.min[0], this.max[0]);
      return denormalizedTensor;
    }
  }

  _normalize(tensor) {
    tensor.print();
    const firstTime = this.min === undefined;
    if(firstTime) {
      this.min = [];
      this.max = [];
    }
    let normalizedTensor;

    const featureDimensions = tensor.shape.length > 1 && tensor.shape[1];
    // More than one feature?
    if (featureDimensions && featureDimensions > 1) {
      if(firstTime) {
        this.min.length = featureDimensions;
        this.max.length = featureDimensions;
      }

      // Split into separate 1d tensors
      const features = tf.split(tensor, featureDimensions, 1);

      // Normalize and find min/max values for each feature
      const normalizedFeatures = features.map((featureTensor, i) => {
        if(firstTime) {
          this.min[i] = featureTensor.min();
          this.max[i] = featureTensor.max();
        }
        return this._minMaxNormalize(featureTensor, this.min[i], this.max[i]);
      });

      features.forEach(t => t.dispose());

      normalizedTensor = tf.concat(normalizedFeatures, 1);
    }
    else {
      if(firstTime) {
        this.min.push(tensor.min());
        this.max.push(tensor.max());
      }
      normalizedTensor = this._minMaxNormalize(tensor, this.min[0], this.max[0]);
    }
    return normalizedTensor;
  }

  // min/max normailization
  _minMaxNormalize(tensor1d, min, max) {
    return tensor1d.sub(min).div(max.sub(min));
  }

  // min/max denormalization
  _minMaxDenormalize(tensor1d, min, max) {
    return tensor1d.mul(max.sub(min)).add(min);
  }

}