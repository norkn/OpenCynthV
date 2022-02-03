let context = new AudioContext();
let masterGain = context.createGain();
let finalGraph;



function update(){

    fetch('/update')
        .then(()=>{
            fetch('/get_graph')
                .then((response)=>{ 
                    //build webaudio graph
                    response.json().then((graph_json) => {
                        
                        
                        finalGraph = JSON.parse(graph_json)
                        console.log("test " + finalGraph);
                    });
                });
    });
};

var intervalId = setInterval(update, 100);    


function logGraph(){
    console.log(this.finalGraph);
};



document.querySelector("#masterGainSlider").addEventListener("input", function(e) {
    masterGain.value = this.value;
    document.querySelector("#masterGainOutput").innerHTML = "Volume " + this.value;
});

document.querySelector("#osciFrequencySlider").addEventListener("input", function(e) {
    document.querySelector("#osciFrequencyOutput").innerHTML = "Frequency: " + this.value;
});

document.querySelector("#panningSlider").addEventListener("input", function (e) {
        
        document.querySelector("#panningOutput").innerHTML = (this.value - 50) / 50 + " LR";
        //stereoPanner.pan.value = panValue;
    });


document.querySelector("#distortionSlider").addEventListener("input", function() {
    document.querySelector("#distortionOutput").innerHTML = this.value;
    //distortion.curve = makeDistortionCurve(this.value);
});


function newWaveShaper(){
    distortion = context.createWaveShaper();
    distortion.curve = makeDistortionCurve(document.querySelector("#distortionSlider").value); 

};

function makeDistortionCurve(amount) {    
    let n_samples = 44100,
        curve = new Float32Array(n_samples);
    
    for (var i = 0; i < n_samples; ++i ) {
        var x = i * 2 / n_samples - 1;
        curve[i] = (Math.PI + amount) * x / (Math.PI + (amount * Math.abs(x)));
    }
    
    return curve;
};


function newOscillator(connectionNode){
    
    let osci = context.createOscillator();
    osci.type = document.querySelector("#osciSelectList").value;
    osci.frequency = document.querySelector("#osciFrequencySlider").value;
    osci.connect(connectionNode);

};


function newPanner(connectionNode){
    pan = context.createStereoPanner();
    pan.value = ((document.querySelector("#panningOutput").value - 50) / 50);
    pan.connect(connectionNode);
}


function buildAudioGraph(){



    masterGain.connect(context.destination);
}