import * as tf from '@tensorflow/tfjs';

export default class Knn {

  /**
   * Constructor
   * @param {tf.tensor2d} features - (e.g.,  long, lat, sqft_living, sqft_lot, )
   * @param {tf.tensor2d} labels - house prices
   */
  constructor(features, labels) {
    // Mean and standard deviation of the features
    const { mean, variance } = tf.moments(features, 0);
    this.mean = mean;
    this.stdDev = variance.pow(0.5);

    // Standarize features using mean and standard deviation
    this.scaledFeatures = features.sub(mean).div(this.stdDev);

    this.labels = labels;
  }


  /**
   * K nearest neighbor prediction
   * @param {tf.tensor} predictionPoint
   * @param {tf.tensor} k - number of nearest neighboors to average
   * @returns {number} - e.g., house price
   */
  predict(predictionPoint, k) {

    const scaledPredictionPoint = predictionPoint.sub(mean).div(variance.pow(0.5));

    return this.scaledFeatures
      .sub(scaledPredictionPoint)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(this.labels, 1)
      .unstack()
      .sort((a,b) => a.get(0) > b.get(0) ? 1 : -1)
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.get(1), 0) / k;
  }
}