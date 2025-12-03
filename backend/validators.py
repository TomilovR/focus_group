import re
from typing import Optional
from config import MAX_SUBJECT_LENGTH, MAX_BODY_LENGTH, MAX_CTA_LENGTH, logger

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class EmailValidator:
    """Validator for email draft inputs"""
    
    # Pattern for detecting potential XSS/injection
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',  # onclick, onload, etc.
        r'<iframe[^>]*>',
        r'eval\(',
        r'expression\(',
    ]
    
    @staticmethod
    def sanitize_text(text: str, max_length: int) -> str:
        """Sanitize and trim text input"""
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Trim whitespace
        text = text.strip()
        
        # Limit length
        if len(text) > max_length:
            logger.warning(f"Text truncated from {len(text)} to {max_length} chars")
            text = text[:max_length]
        
        return text
    
    @staticmethod
    def check_dangerous_content(text: str) -> None:
        """Check for potentially dangerous content"""
        for pattern in EmailValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                raise ValidationError(f"Text contains potentially dangerous content")
    
    @staticmethod
    def validate_subject(subject: str) -> str:
        """Validate and sanitize email subject"""
        if not subject or not subject.strip():
            raise ValidationError("Subject cannot be empty")
        
        sanitized = EmailValidator.sanitize_text(subject, MAX_SUBJECT_LENGTH)
        
        if len(sanitized) < 3:
            raise ValidationError("Subject must be at least 3 characters long")
        
        EmailValidator.check_dangerous_content(sanitized)
        logger.info(f"Subject validated: {len(sanitized)} chars")
        return sanitized
    
    @staticmethod
    def validate_body(body: str) -> str:
        """Validate and sanitize email body"""
        if not body or not body.strip():
            raise ValidationError("Body cannot be empty")
        
        sanitized = EmailValidator.sanitize_text(body, MAX_BODY_LENGTH)
        
        if len(sanitized) < 10:
            raise ValidationError("Body must be at least 10 characters long")
        
        EmailValidator.check_dangerous_content(sanitized)
        logger.info(f"Body validated: {len(sanitized)} chars")
        return sanitized
    
    @staticmethod
    def validate_cta(cta: Optional[str]) -> Optional[str]:
        """Validate and sanitize call-to-action"""
        if not cta:
            return None
        
        sanitized = EmailValidator.sanitize_text(cta, MAX_CTA_LENGTH)
        
        if sanitized:
            EmailValidator.check_dangerous_content(sanitized)
            logger.info(f"CTA validated: {len(sanitized)} chars")
        
        return sanitized if sanitized else None
    
    @staticmethod
    def validate_audience(audience: str) -> str:
        """Validate audience identifier"""
        if not audience or not audience.strip():
            raise ValidationError("Audience cannot be empty")
        
        # Sanitize and limit length
        sanitized = EmailValidator.sanitize_text(audience, 100)
        
        # Check for valid format (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', sanitized):
            raise ValidationError("Audience must contain only alphanumeric characters, hyphens, and underscores")
        
        logger.info(f"Audience validated: {sanitized}")
        return sanitized
