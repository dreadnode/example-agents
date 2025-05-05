<?php
/**
 * Support Sidebar Metabox
 * Settings > Widget Options
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       4.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Metabox for Support
 *
 * @since 4.0
 * @return void
 */
if( !function_exists( 'widgetopts_settings_sidebar_opt_in' ) ):
	function widgetopts_settings_sidebar_opt_in(){ ?>
		<div id="widgetopts-sidebar-widget-optin" class="postbox widgetopts-sidebar-widget">
			<h3 class="hndle ui-sortable-handle"><span><?php _e( 'Download our Ultimate Guide: Optimizing WordPress Sidebars and Widgets for Better Traffic and Conversions', 'widget-options' );?></span></h3>
			<div class="inside">
				<p style="text-align: center;"><img src="<?php echo WIDGETOPTS_PLUGIN_URL . '/assets/images/cover-book.jpg'?>" width="180"></p>
				<form action="https://phpbits.us12.list-manage.com/subscribe/post?u=5597485458f20da5305e44c55&amp;id=6729a2bb96" method="post" class="validate widgetopts-ebook-form" target="_blank" novalidate>
					<p>
						<?php _e( 'Get detailed insights based on our research on 70+ popular websites that will help you optimize your WordPress sidebars and widgets.', 'widget-options' );?>
					</p>
					<p>
						<?php _e( 'Email Address:', 'widget-options' );?><br  />
						<input name="EMAIL" class="required email widefat" placeholder="<?php _e( 'email@domain.com', 'widget-options' );?>" />

						<input type="hidden" name="MMERGE3" value="Freebies" />
						<input type="hidden" name="MMERGE4" value="Plugin Campaign" />
						<input type="hidden" name="MMERGE5" value="Introducing Ultimate Guide: Optimizing WordPress Sidebars and Widgets for Better Traffic and Conversions" />
					</p>
					<p>
						<button class="button-secondary"><?php _e( 'Subscribe & Download', 'widget-options' );?></button>
					</p>
				</form>
			</div>
		</div>
		<script type="text/javascript">
			jQuery( document ).ready( function(){
				jQuery( '.widgetopts-ebook-form' ).on( 'submit', function(e){
					if( jQuery( this ).find( '.required' ).val() == '' ){
						jQuery( this ).find( '.required' ).css({ 'border' : '1px solid red' });
						return false;
					}else{
						jQuery( this ).find( '.required' ).removeAttr('style');
					}
				} );
			} );
		</script>

	    <?php
	}
	//add_action( 'widgetopts_module_sidebar', 'widgetopts_settings_sidebar_opt_in', 20 );
endif;
?>
