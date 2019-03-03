class NeuralNetwork {
  int[] nneurons;
  String[] aneurons;
  int nlayers;
  ArrayList<layer> layers = new ArrayList<layer>();
  ArrayList<neuron> inputLayer;
  ArrayList<neuron> outputLayer;
  Float[][][] weights;
  Float learningrate=1.0;
  Float learningrate_bias=1.0;
  float e = 2.718281828459045;

  NeuralNetwork(int[] nneurons_, String[] aneurons_) {
    nneurons = nneurons_;
    aneurons = aneurons_;
    nlayers = nneurons.length;
    weights = new Float[nlayers-1][][];
    initLayers();
    for (int i = 0; i < weights.length; i++) {
      //nneurons[i]+1 je pre bias
      weights[i] = new Float[nneurons[i]+1][nneurons[i+1]];
    }
    for (int i = 0; i < weights.length; i++) {
      for (int j = 0; j < weights[i].length; j++) {
        for (int k = 0; k < weights[i][j].length; k++) {
          weights[i][j][k] = random(-1, 1);
        }
      }
    }
  }

  void initLayers() {
    for (int i = 0; i < nlayers; i++) {
      layers.add(new layer());
    }
    for (int i = 0; i < layers.size(); i++) {
      layers.get(i).LayerIndex = i;
      layers.get(i).addNeurons(nneurons[i], aneurons[i]);
    }
    outputLayer = layers.get(layers.size()-1).neurons;
    inputLayer = layers.get(0).neurons;
  }

  Float[] Train(Float[] inputs_, Float[] target_) {
    Float[] o = Predict(inputs_);
    
    backpropagate(target_);
    
    //update weights
    
    for (int i = 0; i < weights.length; i++) {
      for (int j = 0; j < weights[i].length-1; j++) {
        for (int k = 0; k < weights[i][j].length; k++) {
          layer clay = layers.get(i);
          weights[i][j][k] -= learningrate*clay.neurons.get(j).output*layers.get(i+1).neurons.get(k).dEdx;
        }
      }
    }

    //update biases

    for (int i = 0; i<weights.length; i++) {
      int j=weights[i].length-1;
      for (int k = 0; k < weights[i][j].length; k++) {
        weights[i][j][k] -= learningrate_bias*layers.get(i+1).neurons.get(k).dEdx;
      }
    }
    
    return o;
  }

  Float[] Predict(Float[] inputs) {
    //add inputs into 1st layer neurons
    for (int i = 0; i<inputLayer.size(); i++) {
      inputLayer.get(i).output = inputs[i];
    }
    Float[] prediction=new Float[nneurons[nlayers-1]];

    for(int i = 1; i < layers.size();i++){
      for(neuron nt : layers.get(i).neurons){
        nt.input=0.0;
        for(neuron nf : layers.get(i-1).neurons){
          nt.input+=nf.output*weights[i-1][nf.myIndex][nt.myIndex];
        }
        nt.output=activation(nt.input,nt.activation);
        if(i==layers.size()-1){
        prediction[nt.myIndex]=nt.output;
        }
      }
    }
    return prediction;
  }

  void backpropagate(Float[] target_) {
    for (int i = layers.size()-1; i>-1; i--) {
      layer curr_l = layers.get(i);
      if (i==layers.size()-1) {
        for (int n = 0; n <curr_l.neurons.size(); n++) {
          neuron curr_n = curr_l.neurons.get(n);
          curr_n.dEdy=curr_n.output-target_[n];
          curr_n.dEdx=dxactivation(curr_n.input, curr_n.activation)*curr_n.dEdy;
        }
      } else {
        for (int n = 0; n <curr_l.neurons.size(); n++) {
          neuron curr_n = curr_l.neurons.get(n);
          curr_n.dEdy=0.0;
          for (int k = 0; k < weights[curr_n.myLayer][curr_n.myIndex].length; k++) {
            curr_n.dEdy+=weights[curr_n.myLayer][curr_n.myIndex][k]*layers.get(curr_n.myLayer+1).neurons.get(k).dEdx;
          }
          curr_n.dEdx=dxactivation(curr_n.input, curr_n.activation)*curr_n.dEdy;
        }
      }
    }
  }

  float activation(float x, String a) {
    switch(a) {
    case "Linear":
      return 0.05*x;
    case "Sigmoid":
      return 1/(1+pow(e, -x*0.5));
    case "Analytic":
      return log(1+pow(e, x))-1;
    case "ReLU":
      if (x>=0) {
        return 0.05*x;
      } else {
        return 0.01*x;
      }
    default: 
      return x;
    }
  }

  float dxactivation(float x, String a) {
    switch(a) {
    case "Linear":
      return 0.05;
    case "Sigmoid":
      return activation(x, a)*(1-activation(x, a));
    case "Analytic":
      return 0.434/(1+pow(e, -x));
    case "ReLU":
      if (x>=0) {
        return 0.05;
      } else {
        return 0.01;
      }
    default: 
      return 1.0;
    }
  }

  void saveWeights() {
    StringList temp=new StringList();
    for (int i =0; i< weights.length; i++) {
      for (int j =0; j< weights[i].length; j++) {
        for (int k =0; k< weights[i][j].length; k++) {
          temp.append(str(weights[i][j][k]));
        }
      }
    }
    saveStrings("weights.txt", temp.array());
  }

  void drawnet(int x, int y, int w, int h, int s, color c1, color c2) {
    //draw weights
    int ln=layers.size()-s;
    Float spacing=min(w/(ln+0.0), 400/1920.0*w);
    for (int i = s; i<layers.size()-1; i++) {
      int n1=layers.get(i+1).neurons.size();
      for (int k=0; k<n1; k++) {
        int n2=layers.get(i).neurons.size();
        for (int j=0; j<n2; j++) {

          color c= lerpColor(c1, c2, weights[i][j][k]/2+0.5);
          stroke(red(c), green(c), blue(c), abs(activation(weights[i][j][k], "Sigmoid")-0.5)*256);
          strokeWeight(abs(activation(weights[i][j][k], "Sigmoid")-0.5)*8/1080.0*h);
          Float xf=w/2+(i-s+1-(ln-s)/2.0)*spacing;
          Float yf=h/2.0+(k-(n1-1)/2.0)*min(h/(n1+0.0), 50);
          Float xw=w/2+(i-s-(ln-s)/2.0)*spacing;
          Float yw=h/2.0+(j-(n2-1)/2.0)*min(h/(n2+0.0), 50);
          line(x+xf, y+yf, x+xw, y+yw);
        }
      }
    }

    for (int i = s; i<layers.size(); i++) {
      int n1=layers.get(i).neurons.size();
      for (int k=0; k<n1; k++) {
        stroke(255);
        strokeWeight(2/1080.0*h);
        fill(layers.get(i).neurons.get(k).output*256);
        Float xf=w/2+(i-s-(ln-s)/2.0)*spacing;
        Float yf=h/2.0+(k-(n1-1)/2.0)*min(h/(n1+0.0), 50);
        ellipse(x+xf, y+yf, min(h/(n1+0.0)-5, 40/1080.0*h), min(h/(n1+0.0)-5, 40/1080.0*h));
      }
    }
  }

  class layer {
    int LayerIndex;
    ArrayList<neuron> neurons = new ArrayList<neuron>();
    void addNeurons(int n, String activation) {
      for (int x=0; x<n; x++) {
        neurons.add(new neuron());
        neuron curr_neuron = neurons.get(x);
        curr_neuron.myLayer = LayerIndex;
        curr_neuron.myIndex = x;
        curr_neuron.activation = activation;
      }
    }
  }

  class neuron {
    float input;
    float output;
    float dEdy;
    float dEdx;
    int myLayer;
    int myIndex;
    float myBias=1.0;
    String activation;
  }
}
