document.addEventListener('DOMContentLoaded', function() {
  // Add folding functionality
  document.querySelectorAll('.folded-output').forEach(details => {
    const summary = document.createElement('summary');
    summary.className = 'folded-summary';
    summary.textContent = 'Click to expand output';
    details.prepend(summary);
  });
});
