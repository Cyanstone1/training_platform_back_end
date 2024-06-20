from django.urls import path
from .views import TextComparison

urlpatterns = [
    path('text/', TextComparison.as_view(p_type=1)),
    path('file/', TextComparison.as_view(p_type=0)),
]