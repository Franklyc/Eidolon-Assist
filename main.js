// Main JavaScript functionality for the Eidolon Assist Homepage

document.addEventListener('DOMContentLoaded', function() {
    // Initialize animations and effects
    initScrollAnimations();
    initDemoControls();
    initCopyButtons();
    initSmoothScrolling();
    initTypingAnimation();
    createRandomFloatingElements();
    
    // Add parallax effect to glow orbs
    window.addEventListener('mousemove', handleParallax);
});

// Apply fade-in animations when elements come into view
function initScrollAnimations() {
    const fadeElements = document.querySelectorAll('.feature-card, .section-title, .hero-content, .hero-image, .demo-container, .requirements-container, .about');
    
    // Create intersection observer
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                // Unobserve after animation is applied
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.15 });
    
    // Observe all elements
    fadeElements.forEach(element => {
        observer.observe(element);
    });
}

// Handle demo section controls to show different steps
function initDemoControls() {
    const demoControls = document.querySelectorAll('.demo-control');
    const demoSteps = document.querySelectorAll('.demo-step');
    
    demoControls.forEach(control => {
        control.addEventListener('click', () => {
            // Get step number
            const stepNumber = control.getAttribute('data-step');
            
            // Remove active class from all controls and steps
            demoControls.forEach(c => c.classList.remove('active'));
            demoSteps.forEach(s => s.classList.remove('active'));
            
            // Add active class to current control and step
            control.classList.add('active');
            document.querySelector(`.step${stepNumber}`).classList.add('active');
        });
    });
}

// Initialize code copy buttons
function initCopyButtons() {
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Find the closest code block
            const codeBlock = button.closest('.code-block');
            const codeText = codeBlock.querySelector('code').innerText;
            
            // Copy to clipboard
            navigator.clipboard.writeText(codeText).then(() => {
                // Show copied feedback
                const originalIcon = button.innerHTML;
                button.innerHTML = '<i class="fas fa-check"></i>';
                
                // Reset after 2 seconds
                setTimeout(() => {
                    button.innerHTML = originalIcon;
                }, 2000);
            });
        });
    });
}

// Add smooth scrolling for anchor links
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                window.scrollTo({
                    top: target.offsetTop - 100,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Parallax effect for glow orbs
function handleParallax(e) {
    const glowOrbs = document.querySelectorAll('.glow-orb');
    
    glowOrbs.forEach((orb, index) => {
        // Different movement amount for each orb
        const moveX = (e.clientX - window.innerWidth / 2) * (0.01 + index * 0.005);
        const moveY = (e.clientY - window.innerHeight / 2) * (0.01 + index * 0.005);
        
        // Apply transform with existing animation
        orb.style.transform = `translate(${moveX}px, ${moveY}px)`;
    });
}

// Generate dynamic text typing animation for assistant bubble
function initTypingAnimation() {
    const assistantBubble = document.querySelector('.chat-bubble.assistant');
    const typingAnimation = assistantBubble.querySelector('.typing-animation');
    
    // Wait 3 seconds then replace typing animation with text
    setTimeout(() => {
        // Create text response
        const textResponse = document.createElement('span');
        textResponse.textContent = '';
        const finalText = "I can see a consistent upward trend in the Q3 data. The graph shows a 27% increase in user engagement compared to Q2.";
        
        // Replace typing animation with text
        assistantBubble.innerHTML = '';
        assistantBubble.appendChild(textResponse);
        
        // Animate text appearance letter by letter
        let i = 0;
        const typeInterval = setInterval(() => {
            if (i < finalText.length) {
                textResponse.textContent += finalText.charAt(i);
                i++;
            } else {
                clearInterval(typeInterval);
            }
        }, 30);
    }, 3000);
}

// Create additional floating elements in the background for visual interest
function createRandomFloatingElements() {
    const container = document.createElement('div');
    container.className = 'floating-elements';
    container.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
        overflow: hidden;
    `;
    
    // Add to body
    document.body.appendChild(container);
    
    // Create 12 random elements
    for (let i = 0; i < 12; i++) {
        const element = document.createElement('div');
        const size = Math.random() * 40 + 10; // 10-50px
        const posX = Math.random() * 100; // 0-100%
        const posY = Math.random() * 100; // 0-100%
        const duration = Math.random() * 30 + 20; // 20-50s
        const delay = Math.random() * -30; // -30-0s
        
        // Random shape (circle or rounded square)
        const borderRadius = Math.random() > 0.5 ? '50%' : '20%';
        
        // Random gradient from our color palette
        const gradients = [
            'linear-gradient(135deg, rgba(74, 134, 232, 0.2), rgba(108, 92, 231, 0.1))',
            'linear-gradient(135deg, rgba(108, 92, 231, 0.2), rgba(0, 210, 211, 0.1))',
            'linear-gradient(135deg, rgba(0, 210, 211, 0.2), rgba(74, 134, 232, 0.1))'
        ];
        
        element.style.cssText = `
            position: absolute;
            top: ${posY}%;
            left: ${posX}%;
            width: ${size}px;
            height: ${size}px;
            border-radius: ${borderRadius};
            background: ${gradients[i % gradients.length]};
            opacity: 0.3;
            filter: blur(2px);
            animation: floatRandom ${duration}s infinite ease-in-out ${delay}s;
        `;
        
        container.appendChild(element);
    }
    
    // Add keyframes for random floating
    const style = document.createElement('style');
    style.textContent = `
        @keyframes floatRandom {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            25% { transform: translate(${randomRange(-100, 100)}px, ${randomRange(-100, 100)}px) rotate(${randomRange(-90, 90)}deg); }
            50% { transform: translate(${randomRange(-100, 100)}px, ${randomRange(-100, 100)}px) rotate(${randomRange(-180, 180)}deg); }
            75% { transform: translate(${randomRange(-100, 100)}px, ${randomRange(-100, 100)}px) rotate(${randomRange(-90, 90)}deg); }
        }
    `;
    
    document.head.appendChild(style);
}

// Helper function for random range
function randomRange(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

