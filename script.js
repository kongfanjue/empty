// DOM Elements
const loginForm = document.getElementById('loginForm');
const emailInput = document.getElementById('email');
const passwordInput = document.getElementById('password');
const togglePasswordBtn = document.getElementById('togglePassword');
const emailError = document.getElementById('emailError');
const passwordError = document.getElementById('passwordError');

// Toggle password visibility
togglePasswordBtn.addEventListener('click', function() {
    const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
    passwordInput.setAttribute('type', type);
    
    // Toggle eye icon
    const eyeIcon = this.querySelector('.eye-icon');
    if (type === 'text') {
        eyeIcon.innerHTML = `
            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
            <line x1="1" y1="1" x2="23" y2="23"></line>
        `;
    } else {
        eyeIcon.innerHTML = `
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
            <circle cx="12" cy="12" r="3"></circle>
        `;
    }
});

// Email validation
function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Password validation
function validatePassword(password) {
    return password.length >= 6;
}

// Show error message
function showError(element, message) {
    element.textContent = message;
    element.parentElement.querySelector('input').style.borderColor = '#ef4444';
}

// Clear error message
function clearError(element) {
    element.textContent = '';
    element.parentElement.querySelector('input').style.borderColor = '';
}

// Real-time validation
emailInput.addEventListener('blur', function() {
    if (this.value && !validateEmail(this.value)) {
        showError(emailError, '请输入有效的邮箱地址');
    } else {
        clearError(emailError);
    }
});

emailInput.addEventListener('input', function() {
    if (emailError.textContent) {
        if (validateEmail(this.value)) {
            clearError(emailError);
        }
    }
});

passwordInput.addEventListener('blur', function() {
    if (this.value && !validatePassword(this.value)) {
        showError(passwordError, '密码长度至少为6位');
    } else {
        clearError(passwordError);
    }
});

passwordInput.addEventListener('input', function() {
    if (passwordError.textContent) {
        if (validatePassword(this.value)) {
            clearError(passwordError);
        }
    }
});

// Form submission
loginForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    let isValid = true;
    
    // Validate email
    if (!emailInput.value) {
        showError(emailError, '请输入邮箱地址');
        isValid = false;
    } else if (!validateEmail(emailInput.value)) {
        showError(emailError, '请输入有效的邮箱地址');
        isValid = false;
    } else {
        clearError(emailError);
    }
    
    // Validate password
    if (!passwordInput.value) {
        showError(passwordError, '请输入密码');
        isValid = false;
    } else if (!validatePassword(passwordInput.value)) {
        showError(passwordError, '密码长度至少为6位');
        isValid = false;
    } else {
        clearError(passwordError);
    }
    
    if (isValid) {
        const loginBtn = document.querySelector('.btn-login');
        loginBtn.classList.add('loading');
        
        // Simulate API call
        try {
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // Success - you would normally redirect here
            loginBtn.classList.remove('loading');
            document.querySelector('.login-card').classList.add('success-animation');
            
            // Show success message (in real app, redirect to dashboard)
            alert('登录成功！');
            
            console.log('Login submitted:', {
                email: emailInput.value,
                password: passwordInput.value,
                remember: document.getElementById('remember').checked
            });
            
        } catch (error) {
            loginBtn.classList.remove('loading');
            alert('登录失败，请重试');
        }
    }
});

// Social login handlers
document.querySelector('.btn-wechat').addEventListener('click', function() {
    console.log('WeChat login clicked');
    alert('微信登录功能开发中...');
});

document.querySelector('.btn-github').addEventListener('click', function() {
    console.log('GitHub login clicked');
    alert('GitHub登录功能开发中...');
});

// Add focus effects
const inputs = document.querySelectorAll('input[type="email"], input[type="password"], input[type="text"]');
inputs.forEach(input => {
    input.addEventListener('focus', function() {
        this.parentElement.querySelector('svg').style.color = '#6366f1';
    });
    
    input.addEventListener('blur', function() {
        this.parentElement.querySelector('svg').style.color = '';
    });
});
