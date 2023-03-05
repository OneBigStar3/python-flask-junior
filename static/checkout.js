function ajaxSubmit(formid) {
    var form = $('#'+formid);
    var url = form.attr('action');
    console.log('form is '+ formid);
    $.ajax({
        type: "POST",
        url: url,
        data: form.serialize(),
        success: function (data) {
            if (data.status == "Success") {
                window.location.href =  data.next_url;
            } else {
                $(".msg").text(data.msg);
            }
        }
    });
}