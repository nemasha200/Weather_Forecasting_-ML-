document.addEventListener("DOMContentLoaded", () => {
  fetch('/api/business')
    .then(res => res.json())
    .then(data => {
      document.getElementById("temperature").innerText = `${data.weather.temperature} °C`;
      document.getElementById("rainfall").innerText = `${data.weather.rainfall} mm`;
      document.getElementById("humidity").innerText = `${data.weather.humidity} %`;
      document.getElementById("wind_speed").innerText = `${data.weather.wind_speed} km/h`;

      // Chart.js
      const ctx = document.getElementById('forecastChart').getContext('2d');
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: data.weather.forecast.days,
          datasets: [
            {
              label: 'Temperature (°C)',
              data: data.weather.forecast.temperature,
              borderColor: 'rgb(37, 99, 235)',
              backgroundColor: 'rgba(37, 99, 235, 0.1)',
              tension: 0.3
            },
            {
              label: 'Rainfall (mm)',
              data: data.weather.forecast.rainfall,
              borderColor: 'rgb(59, 130, 246)',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              tension: 0.3
            }
          ]
        },
        options: {
          responsive: true,
          scales: {
            y: { beginAtZero: true }
          }
        }
      });

      // Recommendations
      const recList = document.getElementById('recommendations');
      data.recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.innerText = rec;
        recList.appendChild(li);
      });
    });
});
