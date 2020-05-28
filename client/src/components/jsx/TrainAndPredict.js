import React from 'reactn';
import styled from 'styled-components';
import styles from '../styles.scss';
import * as tfvis from '@tensorflow/tfjs-vis';
import { Checkmark24, Classification24, PlayFilledAlt24, PauseFilled24, Reset24, Shuffle24, DocumentExport24, DocumentImport24  } from '@carbon/icons-react';
import { WatsonMachineLearning32, Analytics32, Analytics24 } from '@carbon/icons-react';
import { Button, TextInput } from 'carbon-components-react';
import Data from '../tf/Data';
import Model from '../tf/Model';
import Constants, {ProblemType} from '../js/Constants';
import Problem from '../js/Problem';

const Blue = styled.span`
  color: ${styles.blue60};
`
const ButtonStyle = styled(Button)`
  margin-right: 5px;
  white-space: nowrap;
`
const ColumnHeading = styled.div`
  display: flex;
  padding: 16px 0;
  /* text-align: center; */
`
const ColumnLabel = styled.div`
  font-size: x-large;
  padding-top: 8px;
  color: ${styles.blue60};
  margin-left: 5px;
`
const Row = styled.div`
  padding-bottom: 16px;
`
const IconContainer = styled.div`
  position: absolute;
  right: 0;
  padding-right: 8px;
`
const PlayIcon = styled(PlayFilledAlt24)`
  stroke: ${styles.ui01};
  fill: ${styles.ui01};
  display: ${props => props.trainingInprogress ? 'none' : undefined};
`
const PauseIcon = styled(PauseFilled24)`
  stroke: ${styles.ui01};
  fill: ${styles.ui01};
  display: ${props => props.trainingInprogress ? undefined : 'none'};
`
const ResetIcon = styled(Reset24)`
  stroke: ${styles.ui01};
  fill: ${styles.ui01};
`
const ShuffleIcon = styled(Shuffle24)`
  stroke: ${styles.ui01};
  fill: ${styles.ui01};
`
const ClassificationIcon = styled(Classification24)`
  stroke: ${styles.ui01};
  fill: ${styles.ui01};
`
const TestIcon = styled(Checkmark24)`
  stroke: ${styles.ui01};
  fill: ${styles.ui01};
`
const SaveIcon = styled(DocumentExport24)`
  stroke: ${styles.ui01};
  fill: ${styles.ui01};
`
const LoadIcon = styled(DocumentImport24)`
  stroke: ${styles.ui01};
  fill: ${styles.ui01};
`
const PredictIcon = styled(Analytics24)`
  stroke: ${styles.ui01};
  fill: ${styles.ui01};
`
const TrainAndTestIcon = styled(WatsonMachineLearning32)`
  fill: ${styles.blue40};
  stroke-width: 2px;
  stroke: ${styles.blue60};
`
const Predict32Icon = styled(Analytics32)`
  fill: ${styles.blue40};
  stroke-width: 2px;
  stroke: ${styles.blue60};
`
const Input = styled(TextInput)`
  width: 16ch;
`
const Predictions = styled.table`
  margin-top: 16px;

`
const PredictionData = styled.td`
  font-size: x-large;
  color: ${styles.blue60};
  padding-right: 1ch;
  text-align: right;
`
const Td = styled.td`
  width: 170px;
  padding: 5px;
`
const Hightlight = styled.span`
  background: ${styles.ui01};
  border-radius: 15px;
  padding: 0 5px;
`
function Loss({loss}) {
  return <Hightlight>{loss === Constants.LOSS_NOT_SET ? 'not set' : loss.toPrecision(5)}</Hightlight>
}

export default class TrainAndPredict extends React.Component {
  state = {
    trainingStatusCb: () => <Blue>Loading data</Blue>,
    testingStatusCb: () => <Blue>Not yet tested</Blue>,
    trainingDisabled: true,
    testingDisabled: true,
    loadDisabled: true,
    saveDisabled: true,
    predictDisabled: true,
    sqft: 2500,
    price: 600000,
    predictions: [],
  }

  constructor() {
    super();
        this.handlePredictPrice = this.handlePredictPrice.bind(this);
    this.handleLoad = this.handleLoad.bind(this);
    this.handleSave = this.handleSave.bind(this);
    this.handleTest = this.handleTest.bind(this);
    this.handleTrain = this.handleTrain.bind(this);
    this.handleResetModel = this.handleResetModel.bind(this);
    this.handleShuffle = this.handleShuffle.bind(this);
    this.handleSqftChange = this.handleSqftChange.bind(this);
    this.handlePriceChange = this.handlePriceChange.bind(this);
    this.handleClassify = this.handleClassify.bind(this);

    this.loadData().then((points) => {

      this.setState({
        trainingStatusCb: () => <Blue>Not trained</Blue>,
        loadDisabled: false,
        trainingDisabled: false});
    });
  }

  render() {
    return (
      <React.Fragment>
        <ColumnHeading>
          <TrainAndTestIcon aria-label={`Train ${Problem.getModelType()}`}></TrainAndTestIcon>
          <ColumnLabel>{`Train - ${Problem.getModelType()}`}</ColumnLabel>
        </ColumnHeading>
        <Row>
          Training status: {this.state.trainingStatusCb()}
        </Row>
        <Row>
          Testing status: {this.state.testingStatusCb()}
        </Row>
        <Row>
          <ButtonStyle kind={this.global.trainingInprogress ? 'danger' : 'primary'} onClick={this.handleTrain} disabled={this.state.trainingDisabled}>
            Train Model
            <IconContainer>
              <PlayIcon aria-label="Start training" trainingInprogress={this.global.trainingInprogress}/>
              <PauseIcon aria-label="Resume training" trainingInprogress={this.global.trainingInprogress}/>
            </IconContainer>
          </ButtonStyle>
          <ButtonStyle kind="primary" onClick={this.handleTest} disabled={this.state.testingDisabled}>
            Test Model
            <IconContainer>
              <TestIcon aria-label="Test Model"/>
            </IconContainer>
          </ButtonStyle>
        </Row>
        <Row>
          <ButtonStyle kind="primary" onClick={this.handleLoad}  disabled={this.state.loadDisabled  || !this.global.modelSaved}>
            Load Model
            <IconContainer>
              <LoadIcon aria-label="Load Model"/>
            </IconContainer>
          </ButtonStyle>
          <ButtonStyle kind="primary" onClick={this.handleSave} disabled={this.state.saveDisabled}>
            Save Model
            <IconContainer>
              <SaveIcon aria-label="Save Model"/>
            </IconContainer>
          </ButtonStyle>
        </Row>
        <Row>
          <ButtonStyle kind="secondary" onClick={this.handleResetModel} disabled={this.state.testingDisabled}>
            Reset Model
            <IconContainer>
              <ResetIcon aria-label="Reset model weights"/>
            </IconContainer>
          </ButtonStyle>
          <ButtonStyle kind="secondary" onClick={this.handleShuffle} disabled={this.global.trainingInprogress}>
            Shuffle Data
            <IconContainer>
              <ShuffleIcon aria-label="Shuffle data"/>
            </IconContainer>
          </ButtonStyle>
        </Row>
        <ColumnHeading>
          <Predict32Icon aria-label="Predict"></Predict32Icon>
          <ColumnLabel>Predict</ColumnLabel>
        </ColumnHeading>
        <Row>
          <table>
            <tbody>
            <tr>
              <Td>
                <Input id="sqft-input" labelText="Square feet" value={this.state.sqft} type="number" onChange={this.handleSqftChange}/>
              </Td>
              <Td>
                <Input id="price-input" labelText="Price" disabled={this.global.problemType === ProblemType.HOUSE_PRICE} value={this.state.price} type="number" onChange={this.handlePriceChange}/>
              </Td>
            </tr>
            <tr>
              <Td>
                <ButtonStyle kind="primary" onClick={this.handlePredictPrice}
                      disabled={this.state.predictDisabled || this.global.problemType !== ProblemType.HOUSE_PRICE}
                      style={{width: '100%'}}>
                  Predict Price
                  <IconContainer>
                    <PredictIcon aria-label="Predict price"/>
                  </IconContainer>
                </ButtonStyle>
              </Td>
              <Td>
                <ButtonStyle kind="primary" onClick={this.handleClassify}
                  disabled={this.state.predictDisabled || this.global.problemType === ProblemType.HOUSE_PRICE}
                  style={{width: '100%'}}>
                  Predict Class
                  <IconContainer>
                    <ClassificationIcon aria-label="Predict Class"/>
                  </IconContainer>
                </ButtonStyle>
              </Td>
            </tr>
            </tbody>
          </table>
          <Predictions>
            <tbody>
            {this.state.predictions.map(p => (
              <tr>
                  <PredictionData>{p.name}</PredictionData>
                  <PredictionData>{p.value}</PredictionData>
              </tr>
            ))}
            </tbody>
          </Predictions>
        </Row>
      </React.Fragment>
    );
  }

  loadData() {
    return Data.load();
  }

  handleResetModel() {
    Model.create();
  }

  handleShuffle() {
    Data.shuffle();
  }

  async handleTrain() {
    // Pause?
    if(this.global.trainingInprogress) {
      this.setGlobal({trainingInprogress: false});
      return;
    }

    this.disableAllButtons();
    this.setState({
      trainingDisabled: false,
      predictDisabled: false,
      trainingStatusCb: () => <Blue>Training in progress</Blue>
    });
    tfvis.visor().open();
    this.trainingInprogress = true;
    const [trainingLoss, validationLoss] = await Model.train();
    this.trainingInprogress = false;
    this.setState({
      testingDisabled: false,
      saveDisabled: false,
      predictDisabled: false,
      trainingStatusCb: () => (
        <Blue>
          Loss: <Loss loss={trainingLoss}/> Validation loss: <Loss loss={validationLoss}/>
        </Blue>),
      testingStatusCb: () => <Blue>Not yet tested</Blue>,
    });
  }

  async handleTest() {
    this.disableAllButtons();
    this.setState({
      testingStatusCb: () => <Blue>Testing in progress</Blue>
    });
    const loss = await Model.test();
    this.setState({
      trainingDisabled: false,
      saveDisabled: false,
      predictDisabled: false,
      testingStatusCb: () => (
        <Blue>
          Testing set loss: <Loss loss={Number(loss)}/>
        </Blue>
      )
    });
  }

  async handleLoad() {
    this.disableAllButtons();
    const dateSaved = await Model.load();
    this.setState({
      trainingStatusCb: () => <Blue>{`Loaded ${dateSaved}`}</Blue>,
      predictDisabled: false,
      trainingDisabled: false,
      loadDisabled: true,
      testingDisabled: true,
    });
  }

  async handleSave() {
    const dateSaved = await Model.save();
    this.setState({
      saveDisabled: true,
      loadDisabled: false,
    })
    this.setGlobal({modelSaved: true});
  }

  handleSqftChange(e) {
    this.setState({sqft: Number(e.target.value)});
  }

  handlePriceChange(e) {
    this.setState({price: Number(e.target.value)});
  }

  async handlePredictPrice() {
    const prediction = await Model.predict(this.state.sqft);
    if(prediction) {
      this.setState({
        predictions: [prediction]
      });
    }
  }

  async handleClassify() {
    const predictions = await Model.predict(this.state.sqft, this.state.price);
    if(predictions) {
      this.setState({
        predictions
      });
    }
  }

  disableAllButtons() {
    this.setState({
      trainingDisabled: true,
      testingDisabled: true,
      saveDisabled: true,
      loadDisabled: true,
      predictDisabled: true
    });
  }
}