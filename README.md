# tensorflow-stubs: experimental typing stubs for tensorflow

[![Build Status](https://travis-ci.com/rrbutani/tensorflow-stubs.svg?branch=master)](https://travis-ci.com/rrbutani/tensorflow-stubs)

This repository exists for developing [PEP 484](https://www.python.org/dev/peps/pep-0484/)
compatible typing annotations for [tensorflow](https://github.com/tensorflow/tensorflow).

This package is called "tensorflow-stubs" in compliance with [PEP
561](https://www.python.org/dev/peps/pep-0561/). This allows work to be done on the type
annotations from outside the tensorflow library as tensorflow currently does not have
type annotations.

Please note: *This is very much a work in progress*, pull requests are very welcome!

## Vague roadmap

* Build tests using working/verified production code that will enable us more confidence in building our type stubs
* Create stubs for all tensorflow functions, even if these have types that are too generic such as `Any`. We can then create issues when such code is found and make changes to specify the types.
* Make more specific typing with an aim of getting good coverage of all the most important parts (for example the `Tensor` class and any other foundational components)

## Contributors


Tensorflow stubs has been built based on the code contributions of:

* [Janis Lesinskis](https://www.customprogrammingsolutions.com/about/janis-lesinskis)
* [acvander](https://github.com/acvander)
* [Aapeli Vuorinen](https://www.customprogrammingsolutions.com/about/aapeli-vuorinen)
