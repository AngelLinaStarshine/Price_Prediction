$(document).ready(function () {

    $('#get-prediction-btn').on('click', function () {
        $(this).prop('disabled', true).text('Loading...');

        $.ajax({
            url: '/data_analysis',  
            type: 'POST',
            success: function (data) {
       
                $('#get-prediction-btn').prop('disabled', false).text('Get Bitcoin Prediction');
      
                const plotUrl = data.plot_url;
                $('#plot-container').html(`<img src="data:image/png;base64,${plotUrl}" alt="Bitcoin Price Prediction Plot">`);

                let currentWeekTableHtml = '<table><tr><th>Date</th><th>Price (USD)</th></tr>';
                data.current_week_data.forEach(function (entry) {
                    currentWeekTableHtml += `<tr><td>${entry.date}</td><td>${entry.price}</td></tr>`;
                });
                currentWeekTableHtml += '</table>';
                $('#current-week-table').html(currentWeekTableHtml);

                let futureTableHtml = '<table><tr><th>Date</th><th>Predicted Price (USD)</th></tr>';
                data.future_data.forEach(function (entry) {
                    futureTableHtml += `<tr><td>${entry.date}</td><td>${entry.predicted_price}</td></tr>`;
                });
                futureTableHtml += '</table>';
                $('#future-table').html(futureTableHtml);
            },
            error: function (xhr, status, error) {
                alert('Error fetching prediction data');
                $('#get-prediction-btn').prop('disabled', false).text('Get Bitcoin Prediction');
            }
        });
    });
});

$(document).ready(function () {
    // Initially hide the download form
    $('#download-form').hide();

    // Event listener for "Get Bitcoin Prediction" button
    $('#get-prediction-btn').on('click', function () {
        // Simulate an API call to fetch prediction data
        $.ajax({
            url: '/data_analysis', // Endpoint to fetch prediction data
            type: 'POST',
            success: function (response) {
                // Update plot and tables with response data
                const plotUrl = response.plot_url;
                $('#plot-container').html(`<img src="data:image/png;base64,${plotUrl}" alt="Prediction Plot" />`);

                // Display the download form
                $('#download-form').show();
            },
            error: function () {
                alert('Failed to fetch Bitcoin prediction. Please try again.');
            }
        });
    });
});

document.getElementById('download-form').addEventListener('submit', function (e) {
    const keyInput = document.getElementById('sha256-key');
    if (!keyInput.value.trim()) {
        e.preventDefault();
        alert('Please enter a SHA256 key to proceed.');
    }
});

