# API Documentation

## Getting Started

This API provides endpoints for managing user data and authentication.

### Authentication

All endpoints require a valid API key in the header:
```
Authorization: Bearer YOUR_API_KEY
```

### Endpoints

#### GET /users
Returns a list of all users.

#### POST /users
Creates a new user.

#### GET /users/{id}
Returns a specific user by ID.
