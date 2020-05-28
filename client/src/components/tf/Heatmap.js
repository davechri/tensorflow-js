import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import Model from './Model';
import Data from './Data';
import Problem from '../js/Problem';

class Heatmap {
  async plotPredictionHeatmap (name = "Predicted class", size = 400) {
    const [ valuesPromise, xTicksPromise, yTicksPromise ] = tf.tidy(() => {
      const gridSize = 50;
      const predictionColumns = [];
      // Heatmap order is confusing: columns first (top to bottom) then rows (left to right)
      // We want to convert that to a standard cartesian plot so invert the y values
      for (let colIndex = 0; colIndex < gridSize; colIndex++) {
        // Loop for each column, starting from the left
        const colInputs = [];
        const x = colIndex / gridSize;
        for (let rowIndex = 0; rowIndex < gridSize; rowIndex++) {
          // Loop for each row, starting from the top
          const y = (gridSize - rowIndex) / gridSize; // invert
          colInputs.push([x, y]);
        }

        const colPredictions = Model.model.predict(tf.tensor2d(colInputs));
        predictionColumns.push(colPredictions);
      }
      const valuesTensor = tf.stack(predictionColumns);

      const normalizedLabelsTensor = tf.linspace(0, 1, gridSize);
      const xTicksTensor = Data.denormalizeFeature(normalizedLabelsTensor, 0);
      const yTicksTensor = Data.denormalizeFeature(normalizedLabelsTensor.reverse(), 1);

      return [ valuesTensor.array(), xTicksTensor.array(), yTicksTensor.array() ];
    });

    const values = await valuesPromise;
    const xTicks = await xTicksPromise;
    const xTickLabels = xTicks.map(l => (l/1000).toFixed(1)+"k sqft");
    const yTicks = await yTicksPromise;
    const yTickLabels = yTicks.map(l => "$"+(l/1000).toFixed(0)+"k");

    tf.unstack(values, 2).forEach((values, i) => {
      const data = {
        values,
        xTickLabels,
        yTickLabels,
      };

      let className = '';
      if(Problem.isMultiClassClassification()) {
        className = ':' + Data.getClassName(i);
      }

      // tfvis.render.heatmap({
      //   name: `${Problem.getLabel()+className} (unscaled)`,
      //   tab: `Heat map`
      // }, data, { height: size });
      tfvis.render.heatmap({
        name: `${Problem.getLabel()+className}`,
        tab: `Heatmap`
      }, data, { height: size, domain: [0, 1] });
    });
  }
}

export default Heatmap = new Heatmap();