// Landing page JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Animated counters
    const counters = document.querySelectorAll('.counter');
    const counterSpeed = 200; // Speed of counting animation in milliseconds

    counters.forEach(counter => {
        const target = +counter.getAttribute('data-target');
        const count = +counter.innerText;
        const increment = target / (counterSpeed / 10); // Divide animation duration into steps

        if(count < target) {
            const updateCount = () => {
                let currentCount = +counter.innerText;
                if(currentCount < target) {
                    counter.innerText = Math.ceil(currentCount + increment);
                    setTimeout(updateCount, 10);
                } else {
                    counter.innerText = target;
                }
            };
            updateCount();
        }
    });

    // FAQ Accordion
    const faqQuestions = document.querySelectorAll('.faq-question');

    faqQuestions.forEach(question => {
        question.addEventListener('click', () => {
            const faqItem = question.parentElement;
            
            // Toggle active class
            faqItem.classList.toggle('active');
            question.classList.toggle('active');
            
            // Close other FAQ items
            const allFaqItems = document.querySelectorAll('.faq-item');
            allFaqItems.forEach(item => {
                if(item !== faqItem) {
                    item.classList.remove('active');
                    item.querySelector('.faq-question').classList.remove('active');
                }
            });
        });
    });
    
    // Show first FAQ item by default
    if(faqQuestions.length > 0) {
        faqQuestions[0].click();
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            
            if(targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            
            if(targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 100, // Offset for fixed header
                    behavior: 'smooth'
                });
            }
        });
    });

    // Navbar background change on scroll
    const nav = document.querySelector('nav');
    
    window.addEventListener('scroll', () => {
        if(window.scrollY > 50) {
            nav.style.background = 'rgba(0, 0, 0, 0.9)';
            nav.style.backdropFilter = 'blur(10px)';
        } else {
            nav.style.background = 'transparent';
            nav.style.backdropFilter = 'blur(5px)';
        }
    });
    
    // Trigger scroll once to set initial state
    window.dispatchEvent(new Event('scroll'));

    // Add particles to the hero section
    createParticles();
});

// Create particle effect
function createParticles() {
    const particleContainer = document.querySelector('.particle-container');
    
    if(!particleContainer) return;
    
    const particleCount = 50;
    
    for(let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random position
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        
        // Random size
        const size = Math.random() * 5 + 1;
        
        // Random animation duration
        const duration = Math.random() * 20 + 10;
        
        // Random animation delay
        const delay = Math.random() * 10;
        
        // Random opacity
        const opacity = Math.random() * 0.5 + 0.1;
        
        // Apply styles
        particle.style.cssText = `
            position: absolute;
            top: ${posY}%;
            left: ${posX}%;
            width: ${size}px;
            height: ${size}px;
            background: ${Math.random() > 0.5 ? '#5f00ff' : '#00bcd4'};
            border-radius: 50%;
            opacity: ${opacity};
            animation: floatParticle ${duration}s linear infinite;
            animation-delay: -${delay}s;
            pointer-events: none;
        `;
        
        particleContainer.appendChild(particle);
    }
    
    // Add the keyframe animation to the document head
    const styleSheet = document.createElement('style');
    styleSheet.type = 'text/css';
    styleSheet.innerText = `
        @keyframes floatParticle {
            0% {
                transform: translateY(0) translateX(0);
            }
            25% {
                transform: translateY(-20px) translateX(10px);
            }
            50% {
                transform: translateY(0) translateX(20px);
            }
            75% {
                transform: translateY(20px) translateX(10px);
            }
            100% {
                transform: translateY(0) translateX(0);
            }
        }
    `;
    document.head.appendChild(styleSheet);
}
