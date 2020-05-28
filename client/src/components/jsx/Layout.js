import React from 'reactn';
import styled from 'styled-components';
import styles from '../styles.scss';
import { Button, Select, SelectItem } from 'carbon-components-react';
import { OpenPanelRight24  } from '@carbon/icons-react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { Settings32 } from '@carbon/icons-react';
import TrainAndPredict from './TrainAndPredict';
import Settings from './Settings';
import Constants from '../js/Constants';
import Global from '../js/Global';
import Data from '../tf/Data';
import Problem from '../js/Problem';
import {ProblemType} from '../js/Constants';

window.tf = tf; // make tf global so it can be referenced in the browser console

const headerHeight = '100px';
const outerBorder = '32px';
const innerBorder = '16px';

const Container = styled.div`
  width: 100vw;
  height: 100vh;
  padding: ${outerBorder} ${outerBorder};
`
const TitleContainer = styled.div`
  height: ${headerHeight};
`
const Title = styled.div`
  font-size: xx-large;
  font-weight: bold;
`
const ProblemContainer = styled(Select)`
  color: ${styles.blue60};
`
const Body = styled.div`
  padding: ${innerBorder} ${innerBorder};
  background: ${styles.borderColor};
  height: calc(100vh - ${headerHeight} - ${outerBorder} - ${outerBorder});
  overflow-y: auto;
`
const BodyHeader = styled.div`
  height: 40px;
`
const BodyHeaderItem = styled.div`
  float: left;
  margin-right: 16px;
  font-size: x-large;
  line-height: 1.5;
  color: ${styles.blue60};
`
const OpenVisorContainer = styled.div`
  float: right;
`
const OpenVisorButton = styled(Button)`
  border-radius: 15px;
  float: right;
`
const IconContainer = styled.div`
  position: absolute;
  right: 0;
  padding-right: 8px;
`
const OpenVisorIcon = styled(OpenPanelRight24)`
  stroke: ${styles.ui01};
  fill: ${styles.ui01};
`
const LeftColumn = styled.div`
  float: left;
  width: 33%;
  height: 100%;
`
const MiddleColumn = styled.div`
  float: left;
  width: 33%;
  height: 100%;
`
const RightColumn = styled.div`
  float: left;
  width: 33%;
  height: 100%;
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
const Hightlight = styled.span`
  background: ${styles.ui01};
  border-radius: 15px;
  padding: 0 5px;
`
const SettingsIcon = styled(Settings32)`
  fill: ${styles.blue40};
  stroke-width: 2px;
  stroke: ${styles.blue60};
`
function Loss({loss}) {
  return <Hightlight>{loss === Constants.LOSS_NOT_SET ? 'not set' : loss.toPrecision(5)}</Hightlight>
}

export default class Layout extends React.Component {
  state = {
    problemType: this.global.problemType
  }

  constructor() {
    super();
    this.handleOpenVisor = this.handleOpenVisor.bind(this);
  }

  render() {
    return (
      <Container>
        <TitleContainer>
          <Title>
            Analyze Houses Data Set with TensorFlow.js
          </Title>
          <ProblemContainer id="select-problem-type" labelText="" disabled={this.global.trainingInprogress}
            value={this.state.problemType}
            onChange={(e) => this.handleProblemTypeChange(e)}>
            <SelectItem text={'Predict the house price from the living space.'} value={ProblemType.HOUSE_PRICE}/>
            <SelectItem text={'Predict whether or not a house has a water front.'} value={ProblemType.WATER_FRONT}/>
            <SelectItem text={'Predict the number of bedrooms.'} value={ProblemType.BEDROOMS}/>
          </ProblemContainer>
        </TitleContainer>
        <Body>
          <BodyHeader>
            <BodyHeaderItem float="left">
              Seconds: <Hightlight>{this.getTrainingTime()}</Hightlight>
            </BodyHeaderItem>
            <BodyHeaderItem float="left">
              Epochs: <Hightlight>{this.global.currentEpoch}</Hightlight>
            </BodyHeaderItem>
            <BodyHeaderItem float="left">
              Epoch loss: <Loss loss={this.global.currentTrainingLoss}></Loss>
            </BodyHeaderItem>
            <BodyHeaderItem float="left">
              Min loss: <Loss loss={this.global.minTrainingLoss}></Loss>
            </BodyHeaderItem>
            <OpenVisorContainer>
              <OpenVisorButton small kind="primary" onClick={this.handleOpenVisor}>
                Open Visor
                <IconContainer>
                  <OpenVisorIcon aria-label="Open Visor"/>
                </IconContainer>
              </OpenVisorButton>
            </OpenVisorContainer>
          </BodyHeader>
          <div>
            <LeftColumn>
              <TrainAndPredict/>
            </LeftColumn>
            <MiddleColumn>
              <ColumnHeading>
                <SettingsIcon aria-label="Train and Test"></SettingsIcon>
                <ColumnLabel>Settings</ColumnLabel>
              </ColumnHeading>
              <Row>
                <Settings/>
              </Row>
            </MiddleColumn>
            <RightColumn>

            </RightColumn>
          </div>
        </Body>
      </Container>
    );
  }

  getTrainingTime() {
    return this.global.trainingElapsedTime/1000;
  }

  handleProblemTypeChange(e) {
    const problemType = e.target.value;
    this.setState({problemType});
    Global.setProblemType(problemType);

    Problem.change(problemType);

    Data.setupProblem();
  }

  handleOpenVisor() {
    tfvis.visor().open();
  }
}