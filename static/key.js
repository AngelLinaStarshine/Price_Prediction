$(document).ready(function () {
    $('#generate-key').on('click', function () {
        $(this).prop('disabled', true).text('Generating...');  

        $.ajax({
            url: '/generate_sha256_key', 
            type: 'GET',
            success: function (data) {

                $('#generate-key').prop('disabled', false).text('Generate SHA-256 Key');
      
                $('#sha256-key').val(data.key);
                $('#key-display').show();
                $('#proceed-button').show();  
            },
            error: function () {
                alert('Error generating SHA-256 key');
                $('#generate-key').prop('disabled', false).text('Generate SHA-256 Key');
            }
        });
    });

    $('#copy-button').on('click', function () {
        const sha256KeyTextArea = $('#sha256-key')[0];
        sha256KeyTextArea.select();
        document.execCommand('copy');
        Swal.fire({
            title: 'Key Copied!',
            text: 'Your SHA-256 key has been copied to the clipboard.',
            icon: 'success',
            confirmButtonText: 'OK'
        });
    });
});
