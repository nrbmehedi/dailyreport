from django.db import models
from django.core.exceptions import ValidationError

def validate_xlsx(value):
    if not value.name.endswith('.xlsx'):
        raise ValidationError('Only .xlsx files are allowed.')

class TripDataUpload(models.Model):
    FILE_TYPE_CHOICES = [
        ('bid', 'Bid'),
        ('trip', 'Trip'),
        ('subscription', 'Subscription'),
        ('sfq', 'SFQ'),
        ('driver', 'Driver'),
        ('testdata', 'TestData'),
        ('location', 'Location'),
    ]

    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/xlsx/', validators=[validate_xlsx])
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_type = models.CharField(max_length=50, choices=FILE_TYPE_CHOICES)

    def __str__(self):
        return f"{self.name} ({self.file_type})"
