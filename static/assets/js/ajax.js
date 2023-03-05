function rewriter_submit() {
  $("#loading").show();
  var prolonged;
  var lang = document.getElementsByName("lang");
  // Books mode
  var max_length = parseInt(document.getElementById("ngramCount").value);
  var returned_sents = parseInt(document.getElementById("ngram").value);
  var text = document.getElementById("keywords").value;
  var changes = parseInt(document.getElementById('dupValue').value);

  // Blog mode
  var sw_normal=document.getElementById("id-sw-normal").checked;
  var sw_entire=document.getElementById("id-sw-entire").checked;
  var sw_summary=document.getElementById("id-sw-summary").checked;
  var sw_phrase=document.getElementById("id-sw-phrase").checked;
  var sw_mode= parseInt(document.getElementById("sw-mode").value);
  
  for(var i=0; i<lang.length; i++) {
    if(lang[i].checked) var selectedLang = lang[i].value;
  }

 var server_data = [
  {"Text": text},
  {"Max_Length": max_length},
  {"Returned_Sents": returned_sents},
  {"Change_Rate": changes},
  {"Lang": selectedLang},
  {"Sw_Normal" : sw_normal},
  {"Sw_Entire" : sw_entire},
  {"Sw_Summary" : sw_summary},
  {"Sw_Phrase" : sw_phrase},
  {"Sw_Mode" : sw_mode}
 ];
 console.log(server_data)
 $.ajax({
   type: "POST",
   url: "/process_rewrite",
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
  var prolonged;
  var library = document.getElementsByName("select-box1");
  var ngram = parseInt(document.getElementById("ngramCount").value);
  var duplicate = parseInt(document.getElementById("dupValue").value);
  var count = parseInt(document.getElementById("countValue").value);
  var text = document.getElementById("keywords").value;
  var unchanged = document.getElementById("separatewords");
  for(var i=0; i<library.length; i++) {
    if(library[i].checked) var selectedLib = library[i].value;
  }
  
 var server_data = [
  {"Text": text},
  {"Ngram": ngram},
  {"Duplicate": duplicate},
  {"Count": count},
  {"Separate": unchanged},
  {"Lib": selectedLib},
 ];
 $.ajax({
   type: "POST",
   url: "/process_keywords",
   data: JSON.stringify(server_data),
   contentType: "application/json",
   dataType: 'json',
   success: function(result) {
     results.innerHTML = result.words; 
     $("#loading").hide();
   } 
 });
}

