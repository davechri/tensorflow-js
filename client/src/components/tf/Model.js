import {setGlobal, getGlobal} from 'reactn';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import Data from './Data';
import ScatterPlot from './ScatterPlot';
import Constants, {ProblemType} from '../js/Constants';
import Problem from '../js/Problem';
import Heatmap from './Heatmap';
import Weights from '../Weights';

class Model {

  /**
   * Plot orange predition line using 100 example Xs between 0 and 1, or Heatmap
   */
  async plotPrediction() {
    if(getGlobal().problemType === ProblemType.HOUSE_PRICE) {
      tf.tidy(() => {
        const normalizedXs = tf.linspace(0, 1, 100);
        const normalizedYs = this.model.predict(normalizedXs.reshape([100, 1]));
        ScatterPlot.plotPredictionLine(normalizedXs, normalizedYs);
      });
    }
    else {
      await Heatmap.plotPredictionHeatmap();
    }
  }

  /**
   * Create new model
   */
  create() {
    setGlobal({trainingElapsedTime: 0,
                currentEpoch: 0,
                minTrainingLoss: Constants.LOSS_NOT_SET,
                currentTrainingLoss: Constants.LOSS_NOT_SET});

    this.model = tf.sequential({
      layers: [
        // layer 1 (input)
        tf.layers.dense({
          units: getGlobal().hiddenLayers,
          useBias: true,
          activation: getGlobal().activationFunction,
          inputDim: Data.getTrainingFeatureTensor().shape[1]
        }),
        // layer 2 (hidden)
        tf.layers.dense({
          units: getGlobal().hiddenLayers,
          useBias: true,
          activation: getGlobal().activationFunction,
        }),
        // layer 3 (ouptut)
        tf.layers.dense({
          units: Problem.isMultiClassClassification() ? Problem.getClassNames().length : 1,
          useBias: true,
          activation: Problem.isMultiClassClassification() ? 'softmax' : getGlobal().activationFunction,
        })
      ]
    });

    this.compile();

    if(getGlobal().problemType === ProblemType.HOUSE_PRICE) this.plotPrediction();

    this.showModelDetails();
  }

  /**
   * Compile model - set loss and optimization functions
   */
  compile() {
    if(this.model === undefined) {
      this.create();
    }
    else {
      const optimizer = tf.train[getGlobal().optimizer](getGlobal().learningRate);
      this.model.compile({
        loss: getGlobal().lossFunction,
        optimizer,
      });
    }
  }

  fitCallbacks() {
    return tfvis.show.fitCallbacks({ name: "Training Optimization" }, ['loss']);
  }

  showModelDetails() {
    this.fitCallbacks();
    tfvis.show.modelSummary({name: "Model summary"}, this.model);
    for(let i = 0; i < this.model.layers.length; ++i) {
      const layer = this.model.getLayer(undefined, i);
      tfvis.show.layer({name: `Layer ${i+1}`}, layer);
    }
  }

  /**
   * Train model
   * @returns promise([trainingLoss, validationLoss])
   */
  train() {

    if(this.model === undefined) {
      this.create();
    }

    return new Promise(async (resolve) => {
      setGlobal({trainingInprogress: true});
      let epochStartTime = Date.now();

      let bestWeights = Weights.clone(this.model.weights);

      const { onEpochEnd } = this.fitCallbacks(); // draw loss graph

      let callbacks = [];
      callbacks.push({onEpochEnd});
      callbacks.push({
        onEpochEnd: (epoch, logs) => {
          if(logs.loss && logs.loss < getGlobal().minTrainingLoss) {
            setGlobal({minTrainingLoss: logs.loss});
            Weights.copy(bestWeights, this.model.weights);
          }

          setGlobal({currentEpoch: ++getGlobal().currentEpoch,
            currentTrainingLoss: logs.loss,
            trainingElapsedTime:
              getGlobal().trainingElapsedTime + (Date.now() - epochStartTime)});

          epochStartTime = Date.now();

          this.plotPrediction();
        }
      });

      let validationLoss;
      while(getGlobal().trainingInprogress === true) {
        const result = await this.model.fit(Data.getTrainingFeatureTensor(),
          Data.getTrainingLabelTensor(), {
            // initialEpoch: getGlobal().currentEpoch,
            batchSize: getGlobal().batchSize,
            epochs: 1,
            validationSplit: 0.2,
            callbacks
          });

        // console.log(result);
        // console.log(`Training set loss: ${getGlobal().minTrainingLoss}`);
        validationLoss = result.history.val_loss.pop();
        // console.log(`Validation set loss: ${validationLoss}`);
      }

      this.plotPrediction();

      setGlobal({trainingInprogress: false});
      resolve([getGlobal().minTrainingLoss, validationLoss]);
    });
  }

  /**
   * Test model
   * @returns promise([loss])
   */
  async test() {
    return new Promise(async (resolve) => {
      const lossTensor = this.model.evaluate(Data.getTestingFeatureTensor(), Data.getTestingLabelTensor());
      const loss = await lossTensor.data();
      console.log(`Testing set loss: ${loss}`);
      resolve(loss);
    });
  }

  /**
   * Load saved model
   * @returns promise(dateSaved)
   */
  async load() {
    return new Promise(async (resolve) => {
      const storageKey = Problem.getStorageKey();
      const models = await tf.io.listModels();
      const modelInfo = models[storageKey];
      if (modelInfo) {
        this.model = await tf.loadLayersModel(storageKey);

        this.compile();

        this.showModelDetails();

        this.plotPrediction();

        resolve(modelInfo.dateSaved);
      }
      else {
        alert("Could not load: no saved model found");
      }
    });
  }

  /**
   * Save model
   * @returns promise(dateSaved)
   */
  async save() {
    return new Promise(async (resolve) => {
      const storageKey = Problem.getStorageKey();
      const saveResults = await this.model.save(storageKey);
      resolve(saveResults.modelArtifactsInfo.dateSaved);
    })
  }

  /**
   * Is the model saved for the selected problem?
   * @returns promise(boolean) - true, if the model is saved for the selected problem
   */
  isModelSaved() {
    return new Promise(async resolve => {
      const storageKey = Problem.getStorageKey();
      const models = await tf.io.listModels();
      resolve(models[storageKey] !== undefined);
    });
  }

  /**
   * Predict house price
   * @param {*} sqft - sqft
   * @returns promise(price)
   */
  async predict(sqft, price) {
    return new Promise(async (resolve) => {
      if (sqft === undefined || sqft.length === 0 || isNaN(sqft)) {
        alert("Please enter a valid number instead of "+sqft);
        resolve(undefined);
      }
      else {
        if(getGlobal().problemType === ProblemType.HOUSE_PRICE) {
          tf.tidy(async () => {
            const inputTensor = tf.tensor1d([sqft],'int32');
            const normalizedInputTensor = Data.normalizeFeature(inputTensor);
            const normalizedOutputTensor = this.model.predict(normalizedInputTensor);
            const outputTensor = Data.denormalizeLabel(normalizedOutputTensor);
            const outputValue = await outputTensor.data();
            const prediction = `The estimated price is $${(outputValue[0]/1000).toFixed(0)*1000}`;
            resolve({name: 'Price', value: `$${(outputValue[0]/1000).toFixed(0)*1000}`});
          });
        }
        else {
          if (price === undefined || price.length === 0 || isNaN(price)) {
            alert("Please enter a valid number instead of "+price);
            resolve(undefined);
          }
          else {
            tf.tidy(() => {
              const inputTensor = tf.tensor2d([[sqft, price]]);
              const normalizedInput = Data.normalizeFeature(inputTensor);
              const normalizedOutputTensor = this.model.predict(normalizedInput);
              normalizedOutputTensor.print();
              const outputTensor = Data.denormalizeLabel(normalizedOutputTensor);
              const outputValue = outputTensor.dataSync();
              let predictions = [];
              outputValue.forEach((score,i) => {
                const className = Problem.isMultiClassClassification() ? Data.getClassName(i) : '';
                predictions.push({name: `${className} ${Problem.getLabel()}`,
                                  value: `${(score*100).toFixed(1)}%`});
              });
              resolve(predictions);
            });
          }
        }
      }
    });
  }
}

export default Model = new Model();