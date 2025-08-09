document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("contactForm");
  const status = document.getElementById("formStatus");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Simulate form submission
    const name = form.name.value.trim();
    const email = form.email.value.trim();
    const message = form.message.value.trim();

    if (!name || !email || !message) {
      status.textContent = "Please fill in all fields.";
      status.style.color = "red";
      return;
    }

    // Display loading
    status.textContent = "Sending...";
    status.style.color = "black";

    // Simulate async submission
    setTimeout(() => {
      status.textContent = "Message sent successfully! We will get back to you soon.";
      status.style.color = "green";
      form.reset();
    }, 1500);
  });
});
