function myFunction() {
  /* Get the text field */
  var copyText = document.getElementById("results");

  /* Select the text field */
  copyText.select();
  copyText.setSelectionRange(0, 99999); /* For mobile devices */

   /* Copy the text inside the text field */
  navigator.clipboard.writeText(copyText.value);

  var tooltip = document.getElementById("myTooltip");
  tooltip.innerHTML = "Copied!";
}

function outFunc() {
  var tooltip = document.getElementById("myTooltip");
  tooltip.innerHTML = "Copy to clipboard";
}

function clearFunc() {
    var semants = document.getElementById("maxCount");
    semants.remove();
    var semantics = document.getElementById("semResults");
    semantics.remove();
}

var currentdate = new Date(); 
var datetime = "" + currentdate.getDate() + ""
                + (currentdate.getMonth()+1)  + "" 
                + currentdate.getFullYear() + "_"  
                + currentdate.getHours() + ""  
                + currentdate.getMinutes() + "" 
                + currentdate.getSeconds();
function saveTextAsFile() {
  var textToWrite = document.getElementById('results').innerHTML;
  var textFileAsBlob = new Blob([ textToWrite ], { type: 'text/plain' });
  var fileNameToSaveAs = "Export" + datetime + ".txt"; //filename.extension

  var downloadLink = document.createElement("a");
  downloadLink.download = fileNameToSaveAs;
  downloadLink.innerHTML = "Download File";
  if (window.webkitURL != null) {
    // Chrome allows the link to be clicked without actually adding it to the DOM.
    downloadLink.href = window.webkitURL.createObjectURL(textFileAsBlob);
  } else {
    // Firefox requires the link to be added to the DOM before it can be clicked.
    downloadLink.href = window.URL.createObjectURL(textFileAsBlob);
    downloadLink.onclick = destroyClickedElement;
    downloadLink.style.display = "none";
    document.body.appendChild(downloadLink);
  }

  downloadLink.click();
}

var button = document.getElementById('export');
button.addEventListener('click', saveTextAsFile);

function destroyClickedElement(event) {
  // remove the link from the DOM
  document.body.removeChild(event.target);
}


$(document).ready(function(){
    $(".text_div").appendTo("#maxCount");

    if ($('#default').attr("checked")) {
    $('#spaCyLib').show();
    $('#bookBt').removeAttr('required');
    $('#maxCount').show();
    $('#duplicateCount').show();
  }
  if ($('#spaCy').attr("checked")) {
    $('#maxCount').hide();
    $('#duplicateCount').hide();
  }
  if ($('#yake').attr("checked")){
    $('#spaCyLib').hide();
    $('#bookBt').removeAttr('required');
    $('#duplicateCount').show();
    $('#maxCount').show();
  }
  if ($('#rake').attr("checked")){
    $('#spaCyLib').hide();
    $('#bookBt').removeAttr('required');
    $('#duplicateCount').hide();
    $('#maxCount').show();
  }
});

$('.dropdown-el').click(function(e) {
  e.preventDefault();
  e.stopPropagation();
  $(this).toggleClass('expanded');
  $('input[type=radio]').removeAttr('checked');
  $('#'+$(e.target).attr('for')).prop('checked',true);
  var str = $(e.target).attr('for')
  var b_id = '#'+str
  $(b_id).attr("checked", "checked")
  
  
  if ($('#default').attr("checked")) {
    $('#spaCyLib').show();
    $('#bookBt').removeAttr('required');
    $('#maxCount').show();
    $('#duplicateCount').show();
  }
  if ($('#spaCy').attr("checked")) {
    $('#spaCyLib').show();
    $('#bookBt').attr('required','');
    $('#bookBt').prop('checked',true);
    $('#maxCount').hide();
    $('#duplicateCount').hide();
  }
  if ($('#yake').attr("checked")){
    $('#spaCyLib').hide();
    $('#bookBt').removeAttr('required');
    $('#duplicateCount').show();
    $('#maxCount').show();
  }
  if ($('#rake').attr("checked")){
    $('#spaCyLib').hide();
    $('#bookBt').removeAttr('required');
    $('#duplicateCount').hide();
    $('#maxCount').show();
  }

  if ($('#rake').attr("checked")){
    $('#spaCyLib').hide();
    $('#bookBt').removeAttr('required');
    $('#duplicateCount').hide();
    $('#maxCount').show();
  }

  if ($('#base').attr("checked") || $('#bookBt').attr("checked")){
    $('#bookMode').show();
    $('#blogMode').hide(); 
  }

  if ($('#newsBt').attr("checked")){
    $('#bookMode').hide();
    $('#blogMode').show(); 
  }
});

$(document).click(function() {
  $('.dropdown-el').removeClass('expanded');
});


const allRanges = document.querySelectorAll(".range-wrap");
allRanges.forEach(wrap => {
  const range = wrap.querySelector(".range");
  const bubble = wrap.querySelector(".bubble");

  range.addEventListener("input", () => {
    setBubble(range, bubble);
  });
  setBubble(range, bubble);
});

function setBubble(range, bubble) {
if (range.id == 'dupValue'){
  const val = range.value;
  const min = range.min ? range.min : 0;
  const max = range.max ? range.max : 100;
  const newVal = Number(((val - min) * 100) / (max - min));
  let bubbleValue = val/10;
  bubble.innerHTML = bubbleValue;
  console.log(bubbleValue)

  // Sorta magic numbers based on size of the native UI thumb
  bubble.style.left = `calc(${newVal}% + (${8 - newVal * 0.15}px))`;
}
else {
  const val = range.value;
  const min = range.min ? range.min : 0;
  const max = range.max ? range.max : 100;
  const newVal = Number(((val - min) * 100) / (max - min));
  bubble.innerHTML = val;

  // Sorta magic numbers based on size of the native UI thumb
  bubble.style.left = `calc(${newVal}% + (${8 - newVal * 0.15}px))`;
};
  
  
}


$('input[type=range]').on('input', function(e){
  var min = e.target.min,
      max = e.target.max,
      val = e.target.value;
  
  $(e.target).css({
    'backgroundSize': (val - min) * 100 / (max - min) + '% 100%'
  });
}).trigger('input');


// dark-theme UI begin
function saveDarkMode() {	
  var checkbox = document.getElementById("dark-version");
  localStorage.setItem("dark-version", checkbox.checked);
  darkMode();
}
// dark-theme UI end

// navbar minimum begin
function saveNavbarMin() {	
  var checkbox = document.getElementById("navbarMinimize");
  localStorage.setItem("navbarMinimize", checkbox.checked);
  navbarMinimize();
}

// navbar minimum end

// navbar color begin
function saveNavbarColor(a) {	
  var parent = a.parentElement.children;
  var color = a.getAttribute("data-color");

  for (var i = 0; i < parent.length; i++) {
    parent[i].classList.remove('active');
  }

  if (!a.classList.contains('active')) {
    a.classList.add('active');
  } else {
    a.classList.remove('active');
  }

  localStorage.setItem("nav-color", color);
  sidebarColor();
}
// navbar color end

//Set Sidebar Color
function sidebarColor() {
  // var parent = a.parentElement.children;
  // var color = a.getAttribute("data-color");

  // for (var i = 0; i < parent.length; i++) {
  //   parent[i].classList.remove('active');
  // }

  // if (!a.classList.contains('active')) {
  //   a.classList.add('active');
  // } else {
  //   a.classList.remove('active');
  // }
  var color = localStorage.getItem("nav-color");
  var sidebar = document.querySelector('.sidenav');
  sidebar.setAttribute("data-color", color);

  if (document.querySelector('#sidenavCard')) {
    var sidenavCard = document.querySelector('#sidenavCard');
    let sidenavCardClasses = ['card', 'card-background', 'shadow-none', 'card-background-mask-' + color];
    sidenavCard.className = '';
    sidenavCard.classList.add(...sidenavCardClasses);

    var sidenavCardIcon = document.querySelector('#sidenavCardIcon');
    let sidenavCardIconClasses = ['ni', 'ni-diamond', 'text-gradient', 'text-lg', 'top-0', 'text-' + color];
    sidenavCardIcon.className = '';
    sidenavCardIcon.classList.add(...sidenavCardIconClasses);
  }
}

// Set Navbar Minimized
function navbarMinimize() {
  var sidenavShow = document.getElementsByClassName('g-sidenav-show')[0];

  if (JSON.parse(localStorage.getItem("navbarMinimize", false))) {
    sidenavShow.classList.remove('g-sidenav-pinned');
    sidenavShow.classList.add('g-sidenav-hidden');
  } else {
    sidenavShow.classList.remove('g-sidenav-hidden');
    sidenavShow.classList.add('g-sidenav-pinned');
  }
}

// Light Mode / Dark Mode
function darkMode() {
  const body = document.getElementsByTagName('body')[0];
  const hr = document.querySelectorAll('div:not(.sidenav) > hr');
  const sidebar = document.querySelector('.sidenav');
  const sidebarWhite = document.querySelectorAll('.sidenav.bg-white');
  const hr_card = document.querySelectorAll('div:not(.bg-gradient-dark) hr');
  const text_btn = document.querySelectorAll('button:not(.btn) > .text-dark');
  const text_span = document.querySelectorAll('span.text-dark, .breadcrumb .text-dark');
  const text_span_white = document.querySelectorAll('span.text-white, .breadcrumb .text-white');
  const text_strong = document.querySelectorAll('strong.text-dark');
  const text_strong_white = document.querySelectorAll('strong.text-white');
  const text_nav_link = document.querySelectorAll('a.nav-link.text-dark');
  const secondary = document.querySelectorAll('.text-secondary');
  const secondary_light = document.querySelectorAll('.text-white');
  const bg_gray_100 = document.querySelectorAll('.bg-gray-100');
  const bg_gray_600 = document.querySelectorAll('.bg-gray-600');
  const btn_text_dark = document.querySelectorAll('.btn.btn-link.text-dark, .btn .ni.text-dark');
  const btn_text_white = document.querySelectorAll('.btn.btn-link.text-white, .btn .ni.text-white');
  const card_border = document.querySelectorAll('.card.border');
  const card_border_dark = document.querySelectorAll('.card.border.border-dark');
  const svg = document.querySelectorAll('.navbar g');
  const navbarBrand = document.querySelector('.navbar-brand-img');
  const navbarBrandImg = navbarBrand.src;
  const navLinks = document.querySelectorAll('.navbar-main .nav-link, .navbar-main .breadcrumb-item, .navbar-main .breadcrumb-item a');
  const cardNavLinksIcons = document.querySelectorAll('.card .nav .nav-link i');
  const cardNavSpan = document.querySelectorAll('.card .nav .nav-link span');
  const fixedPlugin_blur = document.querySelectorAll('.fixed-plugin > .card');
  const mainContent_blur = document.querySelectorAll('.main-content .container-fluid .card');

  if (JSON.parse(localStorage.getItem("dark-version", false))) {
    body.classList.add('dark-version');
    if (navbarBrandImg.includes('logo-ct-dark.png')) {
      var navbarBrandImgNew = navbarBrandImg.replace("logo-ct-dark", "logo-ct");
      navbarBrand.src = navbarBrandImgNew;
    }

    for (var i = 0; i < mainContent_blur.length; i++) {
      if (mainContent_blur[i].classList.contains('blur')) {
        mainContent_blur[i].classList.remove('blur', 'shadow-blur');
      }
    }
    for (var i = 0; i < navLinks.length; i++) {
      if (navLinks[i].classList.contains('text-white')) {
        navLinks[i].classList.remove('text-white');
      }
    }
    for (var i = 0; i < fixedPlugin_blur.length; i++) {
      if (fixedPlugin_blur[i].classList.contains('blur')) {
        fixedPlugin_blur[i].classList.remove('blur');
      }
    }
    for (var i = 0; i < cardNavLinksIcons.length; i++) {
      if (cardNavLinksIcons[i].classList.contains('text-dark')) {
        cardNavLinksIcons[i].classList.remove('text-dark');
        cardNavLinksIcons[i].classList.add('text-white');
      }
    }
    for (var i = 0; i < cardNavSpan.length; i++) {
      if (cardNavSpan[i].classList.contains('text-sm')) {
        cardNavSpan[i].classList.add('text-white');
      }
    }
    for (var i = 0; i < hr.length; i++) {
      if (hr[i].classList.contains('dark')) {
        hr[i].classList.remove('dark');
        hr[i].classList.add('light');
      }
    }
    for (var i = 0; i < hr_card.length; i++) {
      if (hr_card[i].classList.contains('dark')) {
        hr_card[i].classList.remove('dark');
        hr_card[i].classList.add('light');
      }
    }
    for (var i = 0; i < text_btn.length; i++) {
      if (text_btn[i].classList.contains('text-dark')) {
        text_btn[i].classList.remove('text-dark');
        text_btn[i].classList.add('text-white');
      }
    }
    for (var i = 0; i < text_span.length; i++) {
      if (text_span[i].classList.contains('text-dark')) {
        text_span[i].classList.remove('text-dark');
        text_span[i].classList.add('text-white');
      }
    }
    for (var i = 0; i < text_strong.length; i++) {
      if (text_strong[i].classList.contains('text-dark')) {
        text_strong[i].classList.remove('text-dark');
        text_strong[i].classList.add('text-white');
      }
    }
    for (var i = 0; i < text_nav_link.length; i++) {
      if (text_nav_link[i].classList.contains('text-dark')) {
        text_nav_link[i].classList.remove('text-dark');
        text_nav_link[i].classList.add('text-white');
      }
    }
    for (var i = 0; i < secondary.length; i++) {
      if (secondary[i].classList.contains('text-secondary')) {
        secondary[i].classList.remove('text-secondary');
        secondary[i].classList.add('text-white');
        secondary[i].classList.add('opacity-8');
      }
    }
    for (var i = 0; i < bg_gray_100.length; i++) {
      if (bg_gray_100[i].classList.contains('bg-gray-100')) {
        bg_gray_100[i].classList.remove('bg-gray-100');
        bg_gray_100[i].classList.add('bg-gray-600');
      }
    }
    for (var i = 0; i < btn_text_dark.length; i++) {
      btn_text_dark[i].classList.remove('text-dark');
      btn_text_dark[i].classList.add('text-white');
    }
    for (var i = 0; i < sidebarWhite.length; i++) {
      sidebarWhite[i].classList.remove('bg-white');
    }
    for (var i = 0; i < svg.length; i++) {
      if (svg[i].hasAttribute('fill')) {
        svg[i].setAttribute('fill', '#fff');
      }
    }
    for (var i = 0; i < card_border.length; i++) {
      card_border[i].classList.add('border-dark');
    }
  } else {
    body.classList.remove('dark-version');
    sidebar.classList.add('bg-white');
    if (navbarBrandImg.includes('logo-ct.png')) {
      var navbarBrandImgNew = navbarBrandImg.replace("logo-ct", "logo-ct-dark");
      navbarBrand.src = navbarBrandImgNew;
    }
    for (var i = 0; i < navLinks.length; i++) {
      if (navLinks[i].classList.contains('text-dark')) {
        navLinks[i].classList.add('text-white');
        navLinks[i].classList.remove('text-dark');
      }
    }
    for (var i = 0; i < cardNavLinksIcons.length; i++) {
      if (cardNavLinksIcons[i].classList.contains('text-white')) {
        cardNavLinksIcons[i].classList.remove('text-white');
        cardNavLinksIcons[i].classList.add('text-dark');
      }
    }
    for (var i = 0; i < cardNavSpan.length; i++) {
      if (cardNavSpan[i].classList.contains('text-white')) {
        cardNavSpan[i].classList.remove('text-white');
      }
    }
    for (var i = 0; i < mainContent_blur.length; i++) {
      mainContent_blur[i].classList.add('blur', 'shadow-blur');
    }
    for (var i = 0; i < fixedPlugin_blur.length; i++) {
      fixedPlugin_blur[i].classList.add('blur');
    }
    for (var i = 0; i < hr.length; i++) {
      if (hr[i].classList.contains('light')) {
        hr[i].classList.add('dark');
        hr[i].classList.remove('light');
      }
    }
    for (var i = 0; i < hr_card.length; i++) {
      if (hr_card[i].classList.contains('light')) {
        hr_card[i].classList.add('dark');
        hr_card[i].classList.remove('light');
      }
    }
    for (var i = 0; i < text_btn.length; i++) {
      if (text_btn[i].classList.contains('text-white')) {
        text_btn[i].classList.remove('text-white');
        text_btn[i].classList.add('text-dark');
      }
    }
    for (var i = 0; i < text_span_white.length; i++) {
      if (text_span_white[i].classList.contains('text-white') && !text_span_white[i].closest('.sidenav') && !text_span_white[i].closest('.card.bg-gradient-dark')) {
        text_span_white[i].classList.remove('text-white');
        text_span_white[i].classList.add('text-dark');
      }
    }
    for (var i = 0; i < text_strong_white.length; i++) {
      if (text_strong_white[i].classList.contains('text-white')) {
        text_strong_white[i].classList.remove('text-white');
        text_strong_white[i].classList.add('text-dark');
      }
    }
    for (var i = 0; i < secondary_light.length; i++) {
      if (secondary_light[i].classList.contains('text-white')) {
        secondary_light[i].classList.remove('text-white');
        secondary_light[i].classList.remove('opacity-8');
        secondary_light[i].classList.add('text-secondary');
      }
    }
    for (var i = 0; i < bg_gray_600.length; i++) {
      if (bg_gray_600[i].classList.contains('bg-gray-600')) {
        bg_gray_600[i].classList.remove('bg-gray-600');
        bg_gray_600[i].classList.add('bg-gray-100');
      }
    }
    for (var i = 0; i < svg.length; i++) {
      if (svg[i].hasAttribute('fill')) {
        svg[i].setAttribute('fill', '#6c757d');
      }
    }
    for (var i = 0; i < btn_text_white.length; i++) {
      if (!btn_text_white[i].closest('.card.bg-gradient-dark')) {
        btn_text_white[i].classList.remove('text-white');
        btn_text_white[i].classList.add('text-dark');
      }
    }
    for (var i = 0; i < card_border_dark.length; i++) {
      card_border_dark[i].classList.remove('border-dark');
    }
  }
};