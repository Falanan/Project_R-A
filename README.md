# Project_R-A

## Introduction

In this project, I'm trying to make a video recoder and assistant driving program by using cameras only. Anyway I don't have a car now, Just do some initial experiment now.

## Ideas

My initial idea is firstly make a video recorder, then scaling up more fuctions. All parts are runnable on MacBook. More fuctions I think can be achieved by introducing YOLO model.

## TODOs

* [x] Video Recording - Partically working (Timestep has some issue due to some performance issues)
* [ ] Blind Spot Monitor
* [ ] Approaching Alert
* [ ] Lane Departure Alert
* [ ] Driver fatigue detection


## Genenal Architecture

This program is a Producer-Consumer architecture. All cameras has their own process to recording the video, this is the Producer. There also one process that can plot all cameras at the screen. Then other processes can handle other functions.