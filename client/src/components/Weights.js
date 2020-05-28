import * as tf from '@tensorflow/tfjs'

export default class Weights {

  static copy(toWeights, fromWeights) {
    for(let i = 0; i < fromWeights.length; ++i) {
      // val is an instance of tf.Variable
      toWeights[i].val.assign(fromWeights[i].val);
    }
  }

  static clone(weights) {
    let dupWeights = [];
    for(let i = 0; i < weights.length; ++i) {
      dupWeights.push({val: tf.variable(weights[i].val, true)});
    }
    return dupWeights;
  }


}