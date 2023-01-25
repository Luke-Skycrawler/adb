async function loadData () {
    const words_data = await d3.json('blocks.json');
    return words_data;
}

loadData().then( data =>
    {
        
    });