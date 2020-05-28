import {getGlobal} from 'reactn';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import Data from './Data';
import { assert } from '@tensorflow/tfjs-core/dist/util';
import Problem, {ProblemType} from '../js/Problem';

/**
 * Optional params for regression
 */
export class PlotRegressionParams {
  predictedPointsArray = null;

  constructor(predictedPointsArray = null) {
    this.predictedPointsArray = predictedPointsArray;
  }
}

/**
 * Optional params for classification
 */
export class PlotClassificationParams {
  size = 400;
  equalizeClassSizes = false;

  constructor(size = 400, equalizeClassSizes) {
    this.size = size;
    this.equalizeClassSizes = equalizeClassSizes;
  }
}

class ScatterPlot {

  plot (pointsArray, regressionParams = new PlotRegressionParams(), classificationParams = new PlotClassificationParams()) {
    assert(regressionParams instanceof PlotRegressionParams, 'param 2 expected PlotRegressionParams');
    assert(classificationParams instanceof PlotClassificationParams, 'param 3 expected PlotClassificationParams');

    if(!Problem.isClassification()) {
      const featureName = "Square feet";
      return this.plotRegression(pointsArray, featureName, regressionParams.predictedPointsArray);
    }
    else {
      const classKey = Problem.getLabel();
      return this.plotClasses(pointsArray, classKey, classificationParams.size, classificationParams.equalizeClassSizes);
    }
  }

  async plotRegression (pointsArray, featureName, predictedPointsArray = null) {
    assert(!Problem.isClassification(), 'Problem type must be Regression');

    const values = [pointsArray];
    const series = ["original"];

    if (Array.isArray(predictedPointsArray)) {
      values.push(predictedPointsArray);
      series.push("predicted");
    }

    return tfvis.render.scatterplot(
      { name: `Square feet vs House Price`,
        styles: { width: "100%" }},
      {  values, series },
      {
        xLabel: featureName,
        yLabel: "Price",
      }
    );
  }

  async plotPredictionLine (normalizedXs, normalizedYs) {
    assert(!Problem.isClassification(), 'Problem type must be Regression');

    const [xs, ys] = tf.tidy(() => {

      const xs = Data.denormalizeFeature(normalizedXs);
      const ys = Data.denormalizeLabel(normalizedYs);

      return [ xs.dataSync(), ys.dataSync() ];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: ys[i]}
    });

    await this.plotRegression(Data.points, "Square feet", predictedPoints);
  }

  async plotClasses (pointsArray, classKey, size = 400, equalizeClassSizes = false) {
    assert(Problem.isClassification(), 'Problem type must be Classification');

    // Add each class as a series
    const allSeries = {};
    pointsArray.forEach(p => {
      // Add each point to the series for the class it is in
      const seriesName = `${classKey}: ${p.className}`;
      let series = allSeries[seriesName];
      if (!series) {
        series = [];
        allSeries[seriesName] = series;
      }
      series.push(p);
    });

    if (equalizeClassSizes) {
      // Find smallest class
      let maxLength = null;
      Object.values(allSeries).forEach(series => {
        if (maxLength === null || series.length < maxLength && series.length >= 100) {
          maxLength = series.length;
        }
      });
      // Limit each class to number of elements of smallest class
      Object.keys(allSeries).forEach(keyName => {
        allSeries[keyName] = allSeries[keyName].slice(0, maxLength);
        if (allSeries[keyName].length < 100) {
          delete allSeries[keyName];
        }
      });
    }

    tfvis.render.scatterplot(
      {
        name: `Square feet vs House Price`,
        styles: { width: "100%" }
      },
      {
        values: Object.values(allSeries),
        series: Object.keys(allSeries),
      },
      {
        xLabel: "Square feet",
        yLabel: "Price",
        height: size,
        width: size*1.5,
      }
    );
  }
}

export default ScatterPlot = new ScatterPlot();