from django.db import models

class RegressionModel(models.Model):
    
    reg_task = models.CharField(max_length=100)
    reg_file = models.FileField(upload_to='csv_files/')
    def __str__(self):
        return (
            f"{self.reg_task} ")
    