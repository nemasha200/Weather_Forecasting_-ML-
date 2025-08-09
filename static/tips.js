fetch('/api/tips')
  .then(response => response.json())
  .then(data => {
    // Update weather cards
    document.getElementById('temperature').textContent = `${data.weather.temperature} °C`;
    document.getElementById('rainfall').textContent = `${data.weather.rainfall} mm`;
    document.getElementById('humidity').textContent = `${data.weather.humidity} %`;
    document.getElementById('wind_speed').textContent = `${data.weather.wind_speed} km/h`;

    // Update recommendations
    const recList = document.getElementById('recommendations');
    recList.innerHTML = '';
    data.recommendations.forEach(tip => {
      const li = document.createElement('li');
      li.textContent = tip;
      recList.appendChild(li);
    });

    // Chart 
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
            tension: 0.4
          },
          {
            label: 'Rainfall (mm)',
            data: data.weather.forecast.rainfall,
            borderColor: 'rgb(16, 185, 129)',
            borderDash: [5, 5],
            tension: 0.4
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
  });

// Optional: handle crop select change
document.getElementById('crop-select').addEventListener('change', function() {
  alert(`Switching to tips for: ${this.value}. You can connect this to your API!`);
});
