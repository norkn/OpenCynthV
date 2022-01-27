let graph = '{ "employees" : [' +
'{ "firstName":"John" , "lastName":"Doe" },' +
'{ "firstName":"Anna" , "lastName":"Smith" },' +
'{ "firstName":"Peter" , "lastName":"Jones" } ]}';

function button(){
    console.log("script läuft");
    document.getElementById('test').innerHTML = "script ändert den text";
};
function button2(){
    $.get("/getGraph", function(data){
        graph = $.JSON.parse(data)
        console.log(graph)
    })
};

$(document).ready(function(){

    $('#btnHold').click(function(){

        $.ajax({
          type: 'POST',
          url: '/holdContours',
          success: function(data){
            console.log('konturen erkannt');
            alert(data);
          }
        });
    });

});










let context = new AudioContext();
let gain = context.createGain();
let stereoPanner = context.createStereoPanner();
let delay = context.createDelay(4.0);
let convoler = context.createConvolver();


// Gain, StereoPanner & Delay
document.querySelector("#gainSlider").addEventListener("input", function (e) {
    let gainValue = (this.value / 10);
    document.querySelector("#gainOutput").innerHTML = gainValue + " dB";
    gain.gain.value = gainValue;
});

document.querySelector("#panningSlider").addEventListener("input", function (e) {
    let panValue = ((this.value - 50) / 50);
    document.querySelector("#panningOutput").innerHTML = panValue + " LR";
    stereoPanner.pan.value = panValue;
});




            source.connect(gain);
            gain.connect(delay);
            delay.connect(stereoPanner);
            stereoPanner.connect(compressor);
            compressor.connect(filter);
            filter.connect(distortion);
            distortion.connect(convoler);
            convoler.connect(context.destination);
