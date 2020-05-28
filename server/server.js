'use strict'

const express = require('express');

const app = express();

require('dotenv').config({path:`${process.cwd()}/.env`}); // read .env file

// in production, the client react app is served here
app.use(express.static(`${process.cwd()}/client/build`));

const port = process.env.PORT || 3001;
app.listen(port, async () => {
    console.log('Server listening on port ' + port);
})
