import os
import asyncio
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from loguru import logger
from core.config import settings
from core.celery_app import celery_app

# Define the connection config
if settings.PROD:
    conf = ConnectionConfig(
        MAIL_USERNAME = settings.MAIL_USERNAME,
        MAIL_PASSWORD = settings.MAIL_PASSWORD,
        MAIL_FROM = settings.MAIL_FROM,
        MAIL_PORT = 587,
        MAIL_SERVER = "smtp.gmail.com",
        MAIL_FROM_NAME = "Sheltly Support",
        MAIL_STARTTLS = True, 
        MAIL_SSL_TLS = False,  
        USE_CREDENTIALS = True,
        VALIDATE_CERTS = True
    ) 
else:
    conf = ConnectionConfig(
        MAIL_USERNAME=settings.MAIL_USERNAME,
        MAIL_PASSWORD=settings.MAIL_PASSWORD,
        MAIL_FROM=settings.MAIL_FROM,
        MAIL_PORT=settings.MAIL_PORT,
        MAIL_SERVER=settings.MAIL_SERVER,
        MAIL_FROM_NAME=settings.PROJECT_NAME,
        MAIL_STARTTLS=False, # Set to True for Gmail/Production
        MAIL_SSL_TLS=False,   # Set to True for Port 465
        USE_CREDENTIALS=False, # Set to True for Gmail/Production
        TEMPLATE_FOLDER=os.path.join(os.path.dirname(__file__), "templates/email")
    )
    


@celery_app.task(name="send_verification_email")
def send_verification_email(email_to: str, first_name: str, otp: str):
    """
    Background task to send OTP verification email using fastapi-mail
    """
    logger.info(f"Email verification task started - Recipient: {email_to}, Name: {first_name}")
    
    try:
        message = MessageSchema(
            subject="Sheltly verification code",
            recipients=[email_to],
            template_body={"first_name": first_name, "otp": otp},
            subtype=MessageType.html
        )

        fm = FastMail(conf)
        
        # Since Celery is sync, we run the async send_message in a loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(fm.send_message(message, template_name="email_verification.html"))
        
        logger.success(f"Email verification sent successfully - Recipient: {email_to}")
        return {"status": "success", "recipient": email_to}
    
    except Exception as e:
        logger.error(f"Failed to send verification email to {email_to}: {str(e)}")
        raise


async def send_rejection_email(recipient_email: str, subject: str, message: str):
    """
    Send rejection or notification email asynchronously
    """
    logger.info(f"Sending email to {recipient_email} - Subject: {subject}")
    
    try:
        # Create simple text message
        email_message = MessageSchema(
            subject=subject,
            recipients=[recipient_email],
            body=message,
            subtype=MessageType.plain
        )
        
        fm = FastMail(conf)
        await fm.send_message(email_message)
        
        logger.success(f"Email sent successfully to {recipient_email}")
        return {"status": "success", "recipient": recipient_email}
    
    except Exception as e:
        logger.error(f"Failed to send email to {recipient_email}: {str(e)}")
        # Don't raise, just log - email delivery shouldn't fail the main operation


@celery_app.task(name="send_password_reset_otp")
def send_password_reset_otp(email_to: str, first_name: str, otp: str):
    """
    Background task to send password reset OTP email via fastapi-mail
    """
    logger.info(f"Password reset email task started - Recipient: {email_to}, Name: {first_name}")
    
    try:
        message = MessageSchema(
            subject="Reset your Sheltly password",
            recipients=[email_to],
            template_body={"first_name": first_name, "otp": otp},
            subtype=MessageType.html
        )

        fm = FastMail(conf)
        
        loop = asyncio.get_event_loop()
        loop.run_until_complete(fm.send_message(message, template_name="email_password_reset.html"))
        
        logger.success(f"Password reset email sent successfully - Recipient: {email_to}")
        return {"status": "success", "recipient": email_to}
    
    except Exception as e:
        logger.error(f"Failed to send password reset email to {email_to}: {str(e)}")
        raise


async def send_listing_approved_email(
    recipient_email: str,
    first_name: str,
    property_title: str,
    property_location: str,
    approved_date: str
):
    """
    Send listing approved notification email using template.
    Replaces the old send_approval_notification function.
    """
    logger.info(f"Sending listing approved email to {recipient_email} - Property: {property_title}")
    
    try:
        message = MessageSchema(
            subject="Your Sheltly Listing Has Been Approved!",
            recipients=[recipient_email],
            template_body={
                "first_name": first_name,
                "property_title": property_title,
                "property_location": property_location,
                "approved_date": approved_date
            },
            subtype=MessageType.html
        )
        
        fm = FastMail(conf)
        await fm.send_message(message, template_name="email_listing_approved.html")
        
        logger.success(f"Listing approved email sent successfully to {recipient_email}")
        return {"status": "success", "recipient": recipient_email}
    
    except Exception as e:
        logger.error(f"Failed to send listing approved email to {recipient_email}: {str(e)}")
        # Don't raise - email delivery shouldn't fail the main operation


async def send_listing_rejected_email(
    recipient_email: str,
    first_name: str,
    property_title: str,
    property_location: str,
    rejection_reason: str
):
    """
    Send listing rejected notification email using template.
    Replaces the old send_rejection_notification function.
    Includes the rejection reason visible to the lister.
    """
    logger.info(f"Sending listing rejected email to {recipient_email} - Property: {property_title}")
    
    try:
        message = MessageSchema(
            subject="Sheltly Listing Review Results",
            recipients=[recipient_email],
            template_body={
                "first_name": first_name,
                "property_title": property_title,
                "property_location": property_location,
                "rejection_reason": rejection_reason
            },
            subtype=MessageType.html
        )
        
        fm = FastMail(conf)
        await fm.send_message(message, template_name="email_listing_rejected.html")
        
        logger.success(f"Listing rejected email sent successfully to {recipient_email}")
        return {"status": "success", "recipient": recipient_email}
    
    except Exception as e:
        logger.error(f"Failed to send listing rejected email to {recipient_email}: {str(e)}")
        # Don't raise - email delivery shouldn't fail the main operation