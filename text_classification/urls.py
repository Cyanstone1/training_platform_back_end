from django.urls import path
from .views import *

urlpatterns = [
    path('single_comment/', TextClassification.as_view(p_type=0)),
    path('multiple_comments/', TextClassification.as_view(p_type=1)),
]