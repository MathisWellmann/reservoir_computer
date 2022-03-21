use crate::esn::{ESN, Inputs, Targets};

pub struct FireflyParams {

}

pub struct FireflyOptimizer {
    params: FireflyParams,
}

impl FireflyOptimizer {
    pub fn new(params: FireflyParams) -> Self {
        Self {
            params,
        }
    }

    pub fn step(
        &mut self,
        inputs: &Inputs,
        targets: &Targets,
    ) {
        todo!()
    }

    pub fn elite(&self) -> ESN {
        todo!()
    }
}
