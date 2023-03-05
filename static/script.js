$(document).ready(function() {

    init_first_image();    
    $('#hover_animate_images').hover(
        function(){
            $(this).attr('src', $(this).data('src2'));
        },
        function(){
            $(this).attr('src', $(this).data('src1'));
        }
        );

    if ($("#ajax-form").length) {
        $("#ajax-form").submit(function(event) {
            var form = $(this);
            var url = form.attr('action');
            $.ajax({
                type: "POST",
                url: url,
                data: form.serialize(),
                success: function(data) {
                    if (data == "Success") {
                        window.location.href = "/";
                    } else {
                        $(".msg").text(data);
                    }
                }
            });
            event.preventDefault();
        });
    }
});

function init_first_image(){
    var i = $('#hover_animate_images');
    // now set the images
    i.attr('src', i.data('src1'));
}

function init_second_image(){
    var i = $('#hover_animate_images');
    // set second images
    i.attr('src', i.data('src2'));

}

