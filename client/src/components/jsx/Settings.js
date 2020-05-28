import React from 'reactn';
import styled from 'styled-components';
import styles from '../styles.scss';
import { Select, SelectItem, Slider, TextInput } from 'carbon-components-react';
import Data from '../tf/Data';
import Global from '../js/Global';
import {LossFunction} from '../js/Constants';

const Container = styled.div`
  pointer-events: ${props => props.trainingInprogress ? 'none' : undefined};
`
const Heading = styled.div`
  color: ${styles.blue60};
  margin: 5px 0;
  font-size: large;
`
const SelectStyle = styled(Select)`
  margin-bottom: 5px;
`
const InputStyle = styled(TextInput)`
  width: 14vw;
  margin-bottom: 5px;
`
const SliderStyle = styled(Slider)`
  margin-bottom: 5px;
`
const TrainingTestingSizes = styled.div`
  margin-bottom: 5px;
`
const TrainingTestingSize = styled.span`
  margin-left: 16px;
`
const Iterations = styled.div`
  text-align: center;
  margin-bottom: 5px;
  width: 50%;
`
const Group = styled.div`
  display: flex;
  align-items: center;
  margin-top: 16px;
`
const Input = ({type, label, min, max, value, onChange}) => (
  <InputStyle id={"input-"+value} labelText={label} min={min} max={max} value={value} type={type} onChange={onChange}/>);

export default class Settings extends React.Component {
  render() {
    return (
      <Container trainingInprogress={this.global.trainingInprogress}>
        {/* Creating model */}
        <Heading>Neural network model:</Heading>
        <Group>
          <SelectStyle id="select-activation" labelText="Activation function:"
            value={this.global.activationFunction}
            onChange={(e)=> {Global.setActivationFunction(e.target.value)}}>
            <SelectItem text={'Linear'} value={'linear'}/>
            <SelectItem text={'Relu'} value={'relu'}/>
            <SelectItem text={'Sigmoid'} value={'sigmoid'}/>
            <SelectItem text={'Tanh'} value={'tanh'}/>
          </SelectStyle>
          <Input label="Hidden layers:" min={1} max={100} value={this.global.hiddenLayers} type="number"
            onChange={(e)=> {Global.setHiddenLayers(e.target.value);}}/>
        </Group>
        {/* Training model */}
        <Heading>Training parameters:</Heading>
        <Group>
          <SelectStyle id="select-loss" labelText="Loss function:"
            value={this.global.lossFunction}
            onChange={(e)=> {Global.setLossFunction(e.target.value);}}>
            <SelectItem text={'BinaryCrossentropy'} value={LossFunction.BINARY_CROSS_ENTROPY}/>
            <SelectItem text={'CategoricalCrossentropy'} value={LossFunction.CATEGORICAL_CROSS_ENTROPY}/>
            <SelectItem text={'MeanSquaredError'} value={LossFunction.MEAN_SQUARED_ERROR}/>
          </SelectStyle>
          <Input label="Train while min delta loss is:" min={0} value={this.global.minDeltaLoss} type="number"
              onChange={(e)=> {if(e.target.value > 0) Global.setMinDeltaLoss(e.target.value);}}/>
        </Group>
        <Group>
          <SelectStyle id="select-optimizer" labelText="Optimizer:"
            value={this.global.optimizer}
            onChange={(e)=> {Global.setOptimizer(e.target.value);}}>
            <SelectItem text={'Adam'} value={'adam'}/>
            <SelectItem text={'Stochastic Gradient Descent'} value={'sgd'}/>
            <SelectItem text={'Adagrad'} value={'adagrad'}/>
            <SelectItem text={'Adadelta'} value={'adadelta'}/>
            <SelectItem text={'Adamax'} value={'adamax'}/>
            <SelectItem text={'RMSprop'} value={'rmsprop'}/>
          </SelectStyle>
          <Input label="Learning rate:" min={.001} max={2} value={this.global.learningRate} type="number"
            onChange={(e)=> {Global.setLearningRate(e.target.value);}}/>
        </Group>
        <Group>
          <Input label="Batch size:" min={1} value={this.global.batchSize} type="number"
            onChange={(e)=> {Global.setBatchSize(e.target.value);}}/>
        </Group>
        <this.IterationCount dataSetSize={this.global.dataSetSize}
                              ratio={this.global.trainingToTestingRatio}
                              batchSize={this.global.batchSize}/>
        {/* Data */}
        <Group>
          <SliderStyle labelText="Ratio of training to test data:" min={1} max={99} value={this.global.trainingToTestingRatio}
            onChange={(e)=> {Global.setTrainingToTestingRatio(e.value);}}/>
        </Group>
        <this.TrainingTestingSizes dataSetSize={this.global.dataSetSize} ratio={this.global.trainingToTestingRatio}/>
      </Container>
    );
  }

  IterationCount({dataSetSize,ratio,batchSize}) {
    const trainingSize = Math.round(dataSetSize/(100/ratio));
    const iterationCount = trainingSize/batchSize;
    return (
      <Iterations>Iteratations per epoch: {Math.round(iterationCount)}</Iterations>
    );
  }

  TrainingTestingSizes({dataSetSize, ratio}) {
    const trainingSize = Math.round(dataSetSize/(100/ratio));
    const testingSize = dataSetSize - trainingSize;
    return (
      <TrainingTestingSizes>
        <TrainingTestingSize>Training size: {trainingSize}</TrainingTestingSize>
        <TrainingTestingSize>Testing size: {testingSize}</TrainingTestingSize>
      </TrainingTestingSizes>
    );
  }
}