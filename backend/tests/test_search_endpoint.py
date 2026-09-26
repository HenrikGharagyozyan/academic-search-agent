def test_search_endpoint_returns_results(client, mock_search_service):
    mock_search_service.search.return_value = [
        {"title": "Test Paper", "url": "https://arxiv.org/abs/1", "snippet": "..."}
    ]

    response = client.post("/api/v1/search", json={"query": "gradient descent", "limit": 3})

    assert response.status_code == 200
    assert response.json()["results"][0]["title"] == "Test Paper"