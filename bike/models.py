from django.db import models

class BikeRental(models.Model):
    id = models.AutoField(primary_key=True)
    date = models.DateField()
    location_name = models.CharField(max_length=255, )
    rental_count = models.FloatField()
    return_count = models.FloatField()
    net_change = models.FloatField()
    avg_temp = models.FloatField()
    daily_rainfall = models.FloatField()

    class Meta:
        indexes = [
            models.Index(fields=["date"]),
            models.Index(fields=["location_name", "date"]),
        ]

    def __str__(self):
        return self.location_name