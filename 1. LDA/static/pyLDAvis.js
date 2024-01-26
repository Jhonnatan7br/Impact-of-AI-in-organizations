// Function to toggle visibility of topic keywords
function toggleKeywords() {
    var keywordElements = document.getElementsByClassName('ldavis-topic-term');
    
    for (var i = 0; i < keywordElements.length; i++) {
        var element = keywordElements[i];
        element.style.display = (element.style.display === 'none') ? 'inline-block' : 'none';
    }
}

// Add a button to toggle visibility of topic keywords
var toggleButton = document.createElement('button');
toggleButton.textContent = 'Toggle Keywords';
toggleButton.onclick = toggleKeywords;

// Append the button to the LDA visualization container
var ldaVisContainer = document.getElementById('lda-vis');
ldaVisContainer.appendChild(toggleButton);
