document.addEventListener("DOMContentLoaded", () => {
  const regionSelect = document.getElementById('region');
  const startYearInput = document.getElementById('startYear');
  const endYearInput = document.getElementById('endYear');
  const applyBtn = document.getElementById('applyFilters');
  const insightsBox = document.getElementById('insights');

  let tempChart, rainChart;

  function fetchAndRender() {
    const region = regionSelect.value;
    const startYear = startYearInput.value;
    const endYear = endYearInput.value;

    fetch(`/api/historical?region=${region}&start=${startYear}&end=${endYear}`)
      .then(res => res.json())
      .then(data => {
        const ctxTemp = document.getElementById('temperatureChart').getContext('2d');
        const ctxRain = document.getElementById('rainfallChart').getContext('2d');

        if (tempChart) tempChart.destroy();
        if (rainChart) rainChart.destroy();

        tempChart = new Chart(ctxTemp, {
          type: 'line',
          data: {
            labels: data.years,
            datasets: [{
              label: 'Avg Temperature (°C)',
              data: data.temperature,
              borderColor: 'rgb(147, 51, 234)',
              backgroundColor: 'rgba(14, 170, 160, 0.1)',
              tension: 0.3
            }]
          },
          options: {
            responsive: true,
            scales: { y: { beginAtZero: false } }
          }
        });

        rainChart = new Chart(ctxRain, {
          type: 'bar',
          data: {
            labels: data.years,
            datasets: [{
              label: 'Total Rainfall (mm)',
              data: data.rainfall,
              backgroundColor: 'rgba(168, 85, 247, 0.5)',
              borderColor: 'rgb(168, 85, 247)',
              borderWidth: 1
            }]
          },
          options: {
            responsive: true,
            scales: { y: { beginAtZero: true } }
          }
        });

        insightsBox.innerText = data.insights;
      });
  }

  applyBtn.addEventListener('click', fetchAndRender);

  // Download button
  document.getElementById('downloadBtn').addEventListener('click', () => {
    window.print();
  });

  // Initial load
  fetchAndRender();
});
