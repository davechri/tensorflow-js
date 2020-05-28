import {setGlobal, getGlobal} from 'reactn';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import Problem from '../js/Problem';
import Model from './Model';
import ScatterPlot from './ScatterPlot';
import Normalizer from './Normalizer';
import { assert } from '@tensorflow/tfjs-core/dist/util';

class Data {

  async load() {
    // Import from CSV
    const houseSalesDataset = tf.data.csv('./kc_house_data.csv');

    // Extract x and y values to plot
    const pointsDataset = houseSalesDataset.map(record => ({
      x: record.sqft_living,
      y: record.price,
      className: undefined,
      waterfront: record.waterfront,
      bedrooms: record.bedrooms
    }));

    return pointsDataset.toArray()
    .then((points) => {
      this.points = points;
      this.shuffle();
      return this.points;
    })
  }

  /**
   * Shuffle data
   */
  shuffle() {
    tf.util.shuffle(this.points);
    this.setupProblem();
  }

  /**
   * Do data setup for the selected problem
   */
  async setupProblem() {
    // Clear up tensors for previous problem
    if(this.featureNormalizer) this.featureNormalizer.dispose();
    if(this.labelNormalizer) this.labelNormalizer.dispose();

    tfvis.visor().el.remove();

    // Filter the points (if required) for the select problem
    this.problemPoints = this.points.filter(point => Problem.filterPoint(point));

    ScatterPlot.plot(this.problemPoints);

    setGlobal({dataSetSize: this.problemPoints.length});

    // Update modelSaved boolean for the selected problem
    const modelSaved = await Model.isModelSaved();
    setGlobal({modelSaved});

    // Problem has classes defined?
    if(Problem.isClassification()) {
      this.problemPoints.forEach((p) => {
        p.className = Problem.getClassName(p);
      });
    }

    // Extract Features (inputs)
    // For classification, x,y (sqft,price) are feature, otherwise only sqft is a feature
    const featureValues = this.problemPoints.map(p => !Problem.isClassification() ? p.x : [p.x, p.y]);
    const featureTensor = tf.tensor2d(featureValues, !Problem.isClassification() ? [featureValues.length, 1] : undefined);

    // Extract Labels (outputs)
    // For classification, each class is a label, otherwise the price is the only label
    const labelValues = this.problemPoints.map(p => !Problem.isClassification() ? p.y : this.getClassIndex(p.className));
    // Use one hot encoding if there are more than 2 classes
    const labelTensor = Problem.isMultiClassClassification() ?
            tf.tidy(() => tf.oneHot(tf.tensor1d(labelValues, 'int32'), Problem.getClassNames().length))
            :
            tf.tensor2d(labelValues, [labelValues.length, 1]);
    labelTensor.print();

    // Normalise features and labels
    this.featureNormalizer = new Normalizer(featureTensor);
    this.labelNormalizer = new Normalizer(labelTensor);

    featureTensor.dispose();
    labelTensor.dispose();

    // Default to 50% training to testing ratio
    this.split(getGlobal().trainingToTestingRatio);

    // Create new model and plot data
    Model.create();
    ScatterPlot.plot(this.problemPoints);
  }

  getClassIndex(className) {
    const index = Problem.getClassNames().indexOf(className);
    assert(index !== -1, `getClassIndex: ${index} className ${className} not found: ${Problem.getClassNames()} `);
    return index;
  }

  getClassName(classIndex) {
    assert(classIndex < Problem.getClassNames().length, `getClassName: classIndex ${classIndex} too big: ${Problem.getClassNames()}`);
    return Problem.getClassNames()[classIndex];
  }

  getTrainingFeatureTensor() {
    return this.trainingFeatureTensor;
  }

  getTestingFeatureTensor() {
    return this.testingFeatureTensor;
  }

  getTrainingLabelTensor() {
    return this.trainingLabelTensor;
  }

  getTestingLabelTensor() {
    return this.testingLabelTensor;
  }

  dispose(tensor) {
    if(tensor !== undefined) {
      tensor.dispose();
    }
  }

  split(ratio) {
    this.dispose(this.trainingFeatureTensor);
    this.dispose(this.testingFeatureTensor);
    this.dispose(this.trainingLabelTensor);
    this.dispose(this.testingLabelTensor);

    const trainingSize = Math.round(this.problemPoints.length/(100/ratio));
    const testingSize = this.problemPoints.length - trainingSize;

    const [trainingFeatureTensor, testingFeatureTensor] = tf.split(this.featureNormalizer.getTensor(),
                                                                    [trainingSize, testingSize]);
    const [trainingLabelTensor, testingLabelTensor] = tf.split(this.labelNormalizer.getTensor(),
                                                                [trainingSize, testingSize]);

    this.trainingFeatureTensor = trainingFeatureTensor;
    this.testingFeatureTensor = testingFeatureTensor;
    this.trainingLabelTensor = trainingLabelTensor;
    this.testingLabelTensor = testingLabelTensor;
  }

  normalizeFeature(tensor) {
    return this.featureNormalizer.normalize(tensor);
  }

  /**
   * Denormalize
   * @param {tf.tensor} tensor
   * @param {number} dim - optional dimension to denormalize
   */
  denormalizeFeature(tensor, dim) {
    return this.featureNormalizer.denormalize(tensor, dim);
  }

  denormalizeLabel(tensor) {
    return this.labelNormalizer.denormalize(tensor);
  }
}

export default Data = new Data();