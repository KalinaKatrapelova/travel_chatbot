import requests
import json
import os
from datetime import datetime
import time
from urllib.parse import quote
import random


class DynamicContentProvider:
    """
    Provider for fetching real-time travel-related data.
    This class integrates with various APIs to get weather, events, exchange rates, etc.
    """

    def __init__(self, cache_duration=3600):
        """
        Initialize the dynamic content provider.

        Args:
            cache_duration: Duration in seconds for which data is cached (default: 1 hour)
        """
        self.cache = {}
        self.cache_duration = cache_duration
        self.cache_file = "dynamic_content_cache.json"

        # Load API keys from environment variables or config file
        self.weather_api_key = os.environ.get("WEATHER_API_KEY", "")
        self.event_api_key = os.environ.get("EVENT_API_KEY", "")
        self.currency_api_key = os.environ.get("CURRENCY_API_KEY", "")

        # Load cache from file if it exists
        self._load_cache()

    def _load_cache(self):
        """Load cached data from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.cache = {}

    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _is_cache_valid(self, key):
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False

        cache_time = self.cache[key].get("timestamp", 0)
        current_time = time.time()

        return (current_time - cache_time) < self.cache_duration

    def get_weather(self, city):
        """
        Get current weather information for a city.

        Args:
            city: Name of the city

        Returns:
            dict: Weather information including temperature, conditions, etc.
        """
        cache_key = f"weather_{city.lower()}"

        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]

        # If we don't have an API key, return mock data
        if not self.weather_api_key:
            data = self._get_mock_weather(city)
        else:
            try:
                # Make API request to weather service (OpenWeatherMap example)
                url = f"https://api.openweathermap.org/data/2.5/weather?q={quote(city)}&appid={self.weather_api_key}&units=metric"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    raw_data = response.json()

                    # Process the data into a more usable format
                    data = {
                        "city": city,
                        "temperature": {
                            "current": round(raw_data["main"]["temp"]),
                            "feels_like": round(raw_data["main"]["feels_like"]),
                            "min": round(raw_data["main"]["temp_min"]),
                            "max": round(raw_data["main"]["temp_max"])
                        },
                        "conditions": raw_data["weather"][0]["description"],
                        "humidity": raw_data["main"]["humidity"],
                        "wind_speed": raw_data["wind"]["speed"],
                        "timestamp": datetime.fromtimestamp(raw_data["dt"]).strftime("%Y-%m-%d %H:%M:%S")
                    }
                else:
                    # Fallback to mock data if API fails
                    data = self._get_mock_weather(city)
            except Exception as e:
                print(f"Error fetching weather data: {e}")
                data = self._get_mock_weather(city)

        # Update cache
        self.cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
        self._save_cache()

        return data

    def _get_mock_weather(self, city):
        """Generate mock weather data when API is unavailable"""
        city_temp_ranges = {
            "paris": (10, 25),
            "london": (8, 22),
            "tokyo": (12, 30),
            "new york": (5, 28),
            "rome": (15, 32),
            "barcelona": (15, 28)
        }

        city_lower = city.lower()
        temp_range = city_temp_ranges.get(city_lower, (10, 25))

        current_temp = random.randint(temp_range[0], temp_range[1])

        conditions = random.choice([
            "Clear sky", "Partly cloudy", "Cloudy", "Light rain",
            "Sunny", "Overcast", "Scattered clouds"
        ])

        return {
            "city": city,
            "temperature": {
                "current": current_temp,
                "feels_like": current_temp - random.randint(-2, 2),
                "min": current_temp - random.randint(2, 5),
                "max": current_temp + random.randint(2, 5)
            },
            "conditions": conditions,
            "humidity": random.randint(40, 90),
            "wind_speed": round(random.uniform(1, 15), 1),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "mock data"
        }

    def get_events(self, city, category=None, limit=5):
        """
        Get upcoming events in a city.

        Args:
            city: Name of the city
            category: Type of event (concert, festival, etc.)
            limit: Maximum number of events to return

        Returns:
            list: List of events with details
        """
        cache_key = f"events_{city.lower()}_{category or 'all'}"

        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]

        # For now, use mock data
        # In a real implementation, you'd connect to an events API like Ticketmaster
        data = self._get_mock_events(city, category, limit)

        # Update cache
        self.cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
        self._save_cache()

        return data

    def _get_mock_events(self, city, category=None, limit=5):
        """Generate mock event data"""
        city_lower = city.lower()

        events_by_city = {
            "paris": [
                {"name": "Jazz Night at Le Caveau", "date": "2023-04-15", "category": "music", "price": "€20"},
                {"name": "Louvre Late Night Opening", "date": "2023-04-18", "category": "art", "price": "€12"},
                {"name": "Paris Food Festival", "date": "2023-04-22", "category": "food", "price": "€15"},
                {"name": "Seine River Cruise & Dinner", "date": "2023-04-25", "category": "tour", "price": "€75"},
                {"name": "Montmartre Art Walk", "date": "2023-04-28", "category": "art", "price": "Free"}
            ],
            "london": [
                {"name": "West End Musical Night", "date": "2023-04-16", "category": "theatre", "price": "£35"},
                {"name": "British Museum Special Exhibition", "date": "2023-04-19", "category": "art", "price": "£18"},
                {"name": "Camden Market Food Tour", "date": "2023-04-21", "category": "food", "price": "£25"},
                {"name": "Thames River Boat Party", "date": "2023-04-26", "category": "entertainment", "price": "£45"},
                {"name": "Royal Gardens Tour", "date": "2023-04-29", "category": "tour", "price": "£12"}
            ],
            "tokyo": [
                {"name": "Sakura Festival", "date": "2023-04-15", "category": "festival", "price": "Free"},
                {"name": "Sumo Tournament", "date": "2023-04-20", "category": "sport", "price": "¥3000"},
                {"name": "Anime & Manga Convention", "date": "2023-04-23", "category": "entertainment",
                 "price": "¥2500"},
                {"name": "Traditional Tea Ceremony", "date": "2023-04-26", "category": "cultural", "price": "¥4000"},
                {"name": "Tokyo Tower Night Tour", "date": "2023-04-30", "category": "tour", "price": "¥2000"}
            ],
            "rome": [
                {"name": "Colosseum Night Tour", "date": "2023-04-17", "category": "tour", "price": "€25"},
                {"name": "Italian Opera Night", "date": "2023-04-21", "category": "music", "price": "€40"},
                {"name": "Roman Cuisine Workshop", "date": "2023-04-24", "category": "food", "price": "€55"},
                {"name": "Vatican Museums Special Opening", "date": "2023-04-27", "category": "art", "price": "€20"},
                {"name": "Ancient Rome Walking Tour", "date": "2023-04-30", "category": "tour", "price": "€18"}
            ],
            "barcelona": [
                {"name": "Flamenco Night Show", "date": "2023-04-16", "category": "dance", "price": "€35"},
                {"name": "Gaudi Architecture Tour", "date": "2023-04-20", "category": "tour", "price": "€15"},
                {"name": "Catalan Wine Tasting", "date": "2023-04-22", "category": "food", "price": "€30"},
                {"name": "Mediterranean Cooking Class", "date": "2023-04-25", "category": "food", "price": "€45"},
                {"name": "FC Barcelona Match", "date": "2023-04-28", "category": "sport", "price": "€60+"}
            ]
        }

        # Default events if city not in our mock data
        default_events = [
            {"name": "Local City Tour", "date": "2023-04-15", "category": "tour", "price": "€20"},
            {"name": "Music Festival", "date": "2023-04-20", "category": "music", "price": "€30"},
            {"name": "Food Tasting Event", "date": "2023-04-25", "category": "food", "price": "€25"},
            {"name": "Art Exhibition", "date": "2023-04-28", "category": "art", "price": "€10"},
            {"name": "Cultural Performance", "date": "2023-04-30", "category": "entertainment", "price": "€15"}
        ]

        # Get events for the specified city, or use default events
        city_events = events_by_city.get(city_lower, default_events)

        # Filter by category if specified
        if category:
            city_events = [event for event in city_events if event["category"].lower() == category.lower()]

        # Limit the number of events
        return city_events[:limit]

    def get_exchange_rate(self, from_currency, to_currency):
        """
        Get current exchange rate between two currencies.

        Args:
            from_currency: Source currency code (e.g., 'USD')
            to_currency: Target currency code (e.g., 'EUR')

        Returns:
            float: Exchange rate
        """
        cache_key = f"exchange_{from_currency.upper()}_{to_currency.upper()}"

        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]

        # For now, use mock data
        data = self._get_mock_exchange_rate(from_currency, to_currency)

        # Update cache
        self.cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
        self._save_cache()

        return data

    def _get_mock_exchange_rate(self, from_currency, to_currency):
        """Generate mock exchange rate data"""
        # Common exchange rates (very rough approximations for demonstration)
        rates = {
            "USD_EUR": 0.92,
            "USD_GBP": 0.78,
            "USD_JPY": 110.5,
            "EUR_USD": 1.09,
            "EUR_GBP": 0.85,
            "EUR_JPY": 120.6,
            "GBP_USD": 1.28,
            "GBP_EUR": 1.18,
            "GBP_JPY": 142.1,
            "JPY_USD": 0.0091,
            "JPY_EUR": 0.0083,
            "JPY_GBP": 0.0070
        }

        key = f"{from_currency.upper()}_{to_currency.upper()}"
        rate = rates.get(key, 1.0)  # Default to 1.0 if not found

        # Add a small random variation to make it look more real
        variation = rate * random.uniform(-0.02, 0.02)
        rate += variation

        return {
            "from": from_currency.upper(),
            "to": to_currency.upper(),
            "rate": round(rate, 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "mock data"
        }


# Usage example
if __name__ == "__main__":
    provider = DynamicContentProvider()

    # Get weather for Paris
    weather = provider.get_weather("Paris")
    print(f"Weather in {weather['city']}: {weather['temperature']['current']}°C, {weather['conditions']}")

    # Get events in London
    events = provider.get_events("London", limit=3)
    print("\nUpcoming events in London:")
    for event in events:
        print(f"- {event['name']} ({event['date']}): {event['price']}")

    # Get exchange rate
    exchange = provider.get_exchange_rate("USD", "EUR")
    print(f"\nExchange rate: 1 {exchange['from']} = {exchange['rate']} {exchange['to']}")