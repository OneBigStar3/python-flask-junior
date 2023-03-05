function rewriter_submit() {
  $("#loading").show();
  var prolonged;
  var lang = document.getElementsByName("lang");
  var max_length = parseInt(document.getElementById("ngramCount").value);
  var returned_sents = parseInt(document.getElementById("ngram").value);
  var text = document.getElementById("keywords").value;
  var changes = parseInt(document.getElementById('dupValue').value)
  
  for(var i=0; i<lang.length; i++) {
    if(lang[i].checked) var selectedLang = lang[i].value;
  }

 var server_data = [
  {"Text": text},
  {"Max_Length": max_length},
  {"Returned_Sents": returned_sents},
  {"Change_Rate": changes},
  {"Lang": selectedLang},
 ];
 console.log(server_data)
 $.ajax({
   type: "POST",
   url: "/process_title",
   data: JSON.stringify(server_data),
   contentType: "application/json",
   dataType: 'json',
   success: function(result) {
     results.innerHTML = result.returned; 
     console.log('Returned Result') 
     console.log(result.returned)
     $("#loading").hide();
   } 
 });
}

function keywords_submit() {
  $("#loading").show();
  $("#plots").hide();
  var prolonged;
  var text = document.getElementById("keywords").value;
  var classifier = document.getElementsByName("classifier");
  var d = document.getElementById("demo").innerHTML;
  for(var i=0; i<classifier.length; i++) {
    if(classifier[i].checked) {
      var selectedAlgo = classifier[i].value;
    }
  }
  
 var server_data = [
  {"Text": text},
  {"Algo": selectedAlgo},
  {"d" : d}
 ];
 
 console.log(server_data)
 $.ajax({
   type: "POST",
   url: "/process_title",
   data: JSON.stringify(server_data),
   contentType: "application/json",
   dataType: 'json',
   success: function(result) {
    $("#plots").show();
    results.innerHTML = result.word;
    var source = '/static/images/title_rank_plot.png',
        timestamp = (new Date()).getTime(),
        newUrl = source + '?_=' + timestamp;
    document.getElementById("title_rank").src = newUrl;
    document.getElementById("title_rank").src = '/static/images/title_rank_plot.png';
    console.log(result.vis)
    console.log(result.word)
    $("#loading").hide();
   } 
 });
}

function description_submit() {
  $("#loading").show();
  $("#plots").hide();
  var prolonged;
  var text = document.getElementById("n_samples").value;
  var classifier = document.getElementsByName("algorithm");
  for(var i=0; i<classifier.length; i++) {
    if(classifier[i].checked) {
      var selectedAlgo = classifier[i].value;
    }
  }
  
 var server_data = [
  {"Text": text},
  {"Algo": selectedAlgo},
 ];
 
 console.log(server_data)
 $.ajax({
   type: "POST",
   url: "/process_description",
   data: JSON.stringify(server_data),
   contentType: "application/json",
   dataType: 'json',
   success: function(result) {
    $("#loading").hide();
    console.log(result)
    if (result["returned"].length == 4) {
      for(var i = 0; i < 4; i++){
        document.getElementById("percentage_" + i).innerHTML ="Description Similar Percentage: " + result["returned"][i][0] + "%";
        document.getElementById("title_" + i).innerHTML = "Title: " + result["returned"][i][1];
        document.getElementById("isbn_" + i).innerHTML = "ISBN: " + result["returned"][i][2];
        document.getElementById("description_" + i).innerHTML = "Description: " + result["returned"][i][3];
      }
    }
    $("#plots").show();    
   } 
 });
}