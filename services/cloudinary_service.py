from core.config import settings
import cloudinary
import cloudinary.uploader
import cloudinary.api
from loguru import logger

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True
)


class CloudinaryService:
    """
    Service for handling Cloudinary operations with organized folder structure.
    """
    
    @staticmethod
    def upload_listing_image(file_content: bytes, filename: str, listing_id: str) -> dict:
        """
        Upload a listing image to Cloudinary with optimization transformations.
        Organized in listings/{listing_id}/ folder for clean management.
        
        Args:
            file_content: Image file content as bytes
            filename: Original filename
            listing_id: UUID of the listing for folder organization
            
        Returns:
            Dict with secure_url, public_id, width, height, format, resource_type
        """
        try:
            logger.info(f"Uploading listing image to listings/{listing_id}/ folder")
            
            # Extract filename without extension for clean public_id
            filename_base = filename.rsplit('.', 1)[0] if '.' in filename else filename
            # Clean filename to alphanumeric and underscores only
            filename_clean = "".join(c if c.isalnum() or c == "_" else "_" for c in filename_base)
            
            # Generate unique public_id: timestamp-based to avoid collisions
            import time
            timestamp = int(time.time() * 1000)
            
            folder = f"listings/{listing_id}"
            public_id = f"{filename_clean}_{timestamp}"
            
            # Upload with eager transformation for optimization
            result = cloudinary.uploader.upload(
                file_content,
                folder=folder,
                public_id=public_id,
                resource_type="auto",
                eager=[
                    {
                        "transformation": [
                            {"crop": "limit", "width": 1280, "height": 720},
                            {"quality": "auto", "fetch_format": "auto"}
                        ]
                    }
                ],
                overwrite=False,
                quality="auto",
                fetch_format="auto"
            )
            
            logger.success(f"Listing image uploaded - Folder: listings/{listing_id}/, Public ID: {result.get('public_id')}")
            
            return {
                "secure_url": result.get("secure_url"),
                "public_id": result.get("public_id"),
                "width": result.get("width"),
                "height": result.get("height"),
                "format": result.get("format"),
                "resource_type": result.get("resource_type"),
                "url": result.get("url")
            }
            
        except cloudinary.exceptions.Error as e:
            logger.error(f"Cloudinary upload failed for listing {listing_id}: {str(e)}")
            raise Exception(f"Cloudinary upload failed: {str(e)}")
    
    @staticmethod
    def upload_avatar(file_content: bytes, user_id: str) -> dict:
        """
        Upload a user avatar to Cloudinary with face detection and circular crop.
        Organized in avatars/{user_id}/ folder for clean user-based organization.
        
        Args:
            file_content: Image file content as bytes
            user_id: UUID of the user
            
        Returns:
            Dict with secure_url, public_id, width, height
        """
        try:
            logger.info(f"Uploading avatar to avatars/{user_id}/ folder")
            
            # Delete old avatar if exists
            try:
                old_public_id = f"avatars/{user_id}/avatar"
                cloudinary.uploader.destroy(old_public_id)
                logger.info(f"Old avatar deleted for user: {user_id}")
            except Exception as e:
                logger.warning(f"Could not delete old avatar for user {user_id}: {str(e)}")
            
            # Upload with face detection transformation
            folder = "avatars"
            public_id = user_id  # Use user_id as the public_id for easy lookup
            
            result = cloudinary.uploader.upload(
                file_content,
                folder=f"{folder}/{user_id}",
                public_id="avatar",  # Use consistent name within user folder
                resource_type="auto",
                transformation=[
                    {
                        "crop": "fill",
                        "gravity": "face",
                        "width": 400,
                        "height": 400,
                        "quality": "auto",
                        "radius": "max",
                        "background": "auto"
                    }
                ],
                overwrite=True,
                quality="auto",
                fetch_format="auto"
            )
            
            logger.success(f"Avatar uploaded successfully to avatars/{user_id}/ - Public ID: {result.get('public_id')}")
            
            return {
                "secure_url": result.get("secure_url"),
                "public_id": result.get("public_id"),
                "width": result.get("width"),
                "height": result.get("height"),
                "url": result.get("url")
            }
            
        except cloudinary.exceptions.Error as e:
            logger.error(f"Cloudinary avatar upload failed for user {user_id}: {str(e)}")
            raise Exception(f"Cloudinary upload failed: {str(e)}")
    
    @staticmethod
    def delete_image(public_id: str) -> bool:
        """
        Delete an image from Cloudinary by public_id.
        
        Args:
            public_id: Cloudinary public ID of the image
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Deleting image from Cloudinary: {public_id}")
            
            result = cloudinary.uploader.destroy(public_id)
            
            if result.get("result") == "ok":
                logger.success(f"Image deleted successfully: {public_id}")
                return True
            else:
                logger.warning(f"Image deletion result unclear: {result}")
                return False
                
        except cloudinary.exceptions.Error as e:
            logger.error(f"Failed to delete image {public_id}: {str(e)}")
            raise Exception(f"Failed to delete image: {str(e)}")
    
    @staticmethod
    def validate_file(file_content: bytes, filename: str, file_type: str, max_size_mb: int) -> tuple[bool, str]:
        """
        Validate image file before upload.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            file_type: Expected file type ('listing' or 'avatar')
            max_size_mb: Maximum file size in MB
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            error_msg = f"File too large. Max {max_size_mb}MB allowed"
            logger.warning(f"{error_msg} - Size: {file_size_mb:.2f}MB")
            return False, error_msg
        
        # Check file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        file_ext = f".{filename.split('.')[-1].lower()}"
        
        if file_ext not in allowed_extensions:
            error_msg = "Unsupported file type. Use JPEG, PNG or WebP"
            logger.warning(f"{error_msg} - Extension: {file_ext}")
            return False, error_msg
        
        return True, ""
