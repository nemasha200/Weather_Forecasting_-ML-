document.addEventListener("DOMContentLoaded", function() {

  // ✅ 1️⃣ Show current date & time
  const now = new Date();
  const options = {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  };
  const formattedDateTime = now.toLocaleDateString('en-US', options);
  document.getElementById("currentDateTime").textContent = `Date & Time: ${formattedDateTime}`;

  // ✅ 2️⃣ Fetch weather data & render chart
  fetch('/api/weather')
    .then(response => response.json())
    .then(data => {
      document.getElementById('temperature').textContent = `${data.temperature} °C`;
      document.getElementById('rainfall').textContent = `${data.rainfall} mm`;
      document.getElementById('humidity').textContent = `${data.humidity} %`;
      document.getElementById('wind_speed').textContent = `${data.wind_speed} km/h`;

      const ctx = document.getElementById('forecastChart').getContext('2d');
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.forecast.days,
          datasets: [
            {
              label: 'Temperature (°C)',
              data: data.forecast.temperature,
              borderColor: 'rgb(37, 99, 235)',
              fill: false,
              tension: 0.4
            },
            {
              label: 'Rainfall (mm)',
              data: data.forecast.rainfall,
              borderColor: 'rgb(16, 185, 129)',
              borderDash: [5, 5],
              fill: false,
              tension: 0.4
            }
          ]
        },
        options: {
          responsive: true,
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
      });
    })
    .catch(error => console.error('Error fetching weather data:', error));

  // ✅ 3️⃣ Add to Calendar logic
  const addToCalendarBtn = document.getElementById("addToCalendarBtn");
  if (addToCalendarBtn) {
    addToCalendarBtn.addEventListener("click", function() {
      const eventTitle = encodeURIComponent("Today's Weather Forecast");
      const eventDetails = encodeURIComponent("Check today's weather prediction for agriculture and business planning on the Weather Dashboard.");
      const location = encodeURIComponent("Colombo, Sri Lanka");

      const startDate = now.toISOString().replace(/[-:]/g, '').split('.')[0] + 'Z';
      const endDateObj = new Date(now.getTime() + 60 * 60 * 1000); // +1 hour
      const endDate = endDateObj.toISOString().replace(/[-:]/g, '').split('.')[0] + 'Z';

      const calendarUrl = `https://www.google.com/calendar/render?action=TEMPLATE&text=${eventTitle}&dates=${startDate}/${endDate}&details=${eventDetails}&location=${location}&sf=true&output=xml`;

      window.open(calendarUrl, '_blank');
    });
  }

});
